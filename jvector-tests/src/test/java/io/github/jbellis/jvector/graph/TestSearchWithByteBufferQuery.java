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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class TestSearchWithByteBufferQuery
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void byteBufferQueryMatchesVectorFloatQuery() throws Exception
    {
        int count = 200, dim = 48;
        long seed = 0xFACEB00CL;
        Random r = new Random(seed);

        List<VectorFloat<?>> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dim);
            for (int j = 0; j < dim; j++) v.set(j, r.nextFloat() * 2f - 1f);
            list.add(v);
        }
        var ravv = new ListRandomAccessVectorValues(list, dim);

        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.DOT_PRODUCT);
        try (var builder = new GraphIndexBuilder(bsp, dim, 12, 80, 1.2f, 1.2f, false)) {
            TestUtil.buildSequentially(builder, ravv);
            var graph = builder.getGraph();

            for (int q = 0; q < 10; q++) {
                int qOrd = r.nextInt(count);
                VectorFloat<?> queryAsVectorFloat = list.get(qOrd);

                ByteBuffer queryAsByteBuffer = ByteBuffer.allocate(dim * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
                for (int j = 0; j < dim; j++) queryAsByteBuffer.putFloat(queryAsVectorFloat.get(j));
                queryAsByteBuffer.rewind();

                var resVF = GraphSearcher.search(queryAsVectorFloat, 5, ravv, VectorSimilarityFunction.DOT_PRODUCT, graph, Bits.ALL);
                var resBB = GraphSearcher.search(queryAsByteBuffer, 5, ravv, VectorSimilarityFunction.DOT_PRODUCT, graph, Bits.ALL);

                assertEquals(resVF.getNodes().length, resBB.getNodes().length);
                for (int i = 0; i < resVF.getNodes().length; i++) {
                    assertEquals("node mismatch at i=" + i, resVF.getNodes()[i].node, resBB.getNodes()[i].node);
                    assertEquals("score mismatch at i=" + i, resVF.getNodes()[i].score, resBB.getNodes()[i].score, 1e-5f);
                }
            }
        }
    }

    @Test
    public void addGraphNodeByteBufferMatchesVectorFloat() throws Exception
    {
        int count = 50, dim = 24;
        long seed = 0xC0FFEEL;
        Random r = new Random(seed);
        List<VectorFloat<?>> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dim);
            for (int j = 0; j < dim; j++) v.set(j, r.nextFloat() * 2f - 1f);
            list.add(v);
        }
        var ravvA = new ListRandomAccessVectorValues(list, dim);

        // Build A — using VectorFloat
        var bspA = BuildScoreProvider.randomAccessScoreProvider(ravvA, VectorSimilarityFunction.EUCLIDEAN);
        var builderA = new GraphIndexBuilder(bspA, dim, 10, 60, 1.2f, 1.2f, false);
        for (int i = 0; i < count; i++) builderA.addGraphNode(i, list.get(i));
        builderA.cleanup();

        // Build B — using ByteBuffer overload
        var ravvB = new ListRandomAccessVectorValues(list, dim);
        var bspB = BuildScoreProvider.randomAccessScoreProvider(ravvB, VectorSimilarityFunction.EUCLIDEAN);
        var builderB = new GraphIndexBuilder(bspB, dim, 10, 60, 1.2f, 1.2f, false);
        for (int i = 0; i < count; i++) {
            ByteBuffer bb = ByteBuffer.allocate(dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (int j = 0; j < dim; j++) bb.putFloat(list.get(i).get(j));
            bb.rewind();
            builderB.addGraphNode(i, bb);
        }
        builderB.cleanup();

        TestUtil.assertGraphEquals(builderA.getGraph(), builderB.getGraph());
        builderA.close();
        builderB.close();
    }
}
