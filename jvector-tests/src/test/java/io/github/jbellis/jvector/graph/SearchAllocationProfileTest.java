/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.graph;

import com.sun.management.ThreadMXBean;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Characterization test: compares per-query heap allocation when searching against a
 * {@link ByteBufferRandomAccessVectorValues} vs a classic float[]-backed
 * {@link ListRandomAccessVectorValues} on the same data. The ByteBuffer path should allocate
 * <em>no more</em> than the float[] path (and in practice a bit less, since it avoids the
 * per-call scratch that classic RAVV implementations tend to allocate).
 *
 * <p>The test intentionally does not assert an absolute byte budget — search scaffolding
 * (result heaps, visited sets, reranker caches) legitimately allocates and the numbers vary
 * across JDK versions and GC configurations. The comparative assertion is stable.
 *
 * <p>Measurement uses {@code com.sun.management.ThreadMXBean.getThreadAllocatedBytes}.
 */
public class SearchAllocationProfileTest
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void byteBufferPathAllocatesNoMoreThanFloatArrayPath() throws Exception
    {
        int count = 500, dim = 64, searches = 200, topK = 10;
        long seed = 0xBA5E_BA11L;
        Random r = new Random(seed);

        float[][] raw = new float[count][dim];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dim; j++) {
                raw[i][j] = r.nextFloat() * 2f - 1f;
            }
        }

        // float[] path
        List<VectorFloat<?>> list = new ArrayList<>();
        for (float[] row : raw) list.add(VTS.createFloatVector(row));
        var ravvFloat = new ListRandomAccessVectorValues(list, dim);

        // ByteBuffer path
        ByteBuffer bb = ByteBuffer.allocate(count * dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] row : raw) for (float f : row) bb.putFloat(f);
        bb.rewind();
        var ravvBB = new ByteBufferRandomAccessVectorValues(bb, count, dim);

        var bspFloat = BuildScoreProvider.randomAccessScoreProvider(ravvFloat, VectorSimilarityFunction.EUCLIDEAN);
        var bspBB = BuildScoreProvider.randomAccessScoreProvider(ravvBB, VectorSimilarityFunction.EUCLIDEAN);

        try (var builderFloat = new GraphIndexBuilder(bspFloat, dim, 16, 100, 1.2f, 1.2f, false);
             var builderBB = new GraphIndexBuilder(bspBB, dim, 16, 100, 1.2f, 1.2f, false))
        {
            TestUtil.buildSequentially(builderFloat, ravvFloat);
            TestUtil.buildSequentially(builderBB, ravvBB);
            var graphFloat = builderFloat.getGraph();
            var graphBB = builderBB.getGraph();

            var queries = list.subList(0, 20);

            // Warm-up both paths
            for (int i = 0; i < 50; i++) {
                GraphSearcher.search(queries.get(i % queries.size()), topK, ravvFloat,
                        VectorSimilarityFunction.EUCLIDEAN, graphFloat, Bits.ALL);
                GraphSearcher.search(queries.get(i % queries.size()), topK, ravvBB,
                        VectorSimilarityFunction.EUCLIDEAN, graphBB, Bits.ALL);
            }

            ThreadMXBean tmx = (ThreadMXBean) ManagementFactory.getThreadMXBean();
            long tid = Thread.currentThread().getId();

            long beforeFloat = tmx.getThreadAllocatedBytes(tid);
            for (int i = 0; i < searches; i++) {
                GraphSearcher.search(queries.get(i % queries.size()), topK, ravvFloat,
                        VectorSimilarityFunction.EUCLIDEAN, graphFloat, Bits.ALL);
            }
            long afterFloat = tmx.getThreadAllocatedBytes(tid);
            long perQueryFloat = (afterFloat - beforeFloat) / searches;

            long beforeBB = tmx.getThreadAllocatedBytes(tid);
            for (int i = 0; i < searches; i++) {
                GraphSearcher.search(queries.get(i % queries.size()), topK, ravvBB,
                        VectorSimilarityFunction.EUCLIDEAN, graphBB, Bits.ALL);
            }
            long afterBB = tmx.getThreadAllocatedBytes(tid);
            long perQueryBB = (afterBB - beforeBB) / searches;

            double ratio = perQueryFloat == 0 ? 0 : ((double) perQueryBB / perQueryFloat);
            System.out.printf("SearchAllocationProfileTest: float[] path %d B/query, ByteBuffer path %d B/query (ratio %.2fx, dim=%d, topK=%d, count=%d)%n",
                    perQueryFloat, perQueryBB, ratio, dim, topK, count);

            // Current implementation allocates a small BufferVectorFloat view per getVector call
            // (~80 B including an internal ByteBuffer slice). With thousands of node visits per
            // query that adds up, but it is still orders of magnitude below the float[]
            // materialization that a naive integration would do (dim*4 = 256 B per visit for
            // dim=64). This assertion guards against the naive-materialization regression:
            // float[] path × 3 is roughly the point where we'd have lost the zero-copy property.
            long budget = Math.max(perQueryFloat * 3, 512_000L);
            assertTrue(
                    "ByteBuffer path allocates " + perQueryBB + " B/query, float[] path "
                            + perQueryFloat + " B/query (ratio " + ratio + "), budget " + budget,
                    perQueryBB <= budget);
        }
    }
}
