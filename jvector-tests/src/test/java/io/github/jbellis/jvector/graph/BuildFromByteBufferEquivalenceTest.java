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

/**
 * Proves that building a graph from a caller-owned {@link ByteBuffer} produces the same
 * structure as building from a {@code float[]}-based RAVV using the same seeded data. This
 * is the end-to-end check that the ByteBuffer migration preserves semantics across the
 * full build hot path (VectorFloat abstraction, SIMD dispatch, score provider, diversity
 * pruning, etc.).
 *
 * <p>Runs under whichever vectorization provider is active (Default / Panama / Native)
 * — so the same test catches regressions across all SIMD backends.
 */
public class BuildFromByteBufferEquivalenceTest
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private static float[][] generate(long seed, int count, int dim)
    {
        Random r = new Random(seed);
        float[][] data = new float[count][dim];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dim; j++) {
                data[i][j] = r.nextFloat() * 2f - 1f;
            }
        }
        return data;
    }

    private static ByteBuffer toLittleEndian(float[][] data)
    {
        int count = data.length, dim = data[0].length;
        ByteBuffer bb = ByteBuffer.allocate(count * dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] v : data) for (float f : v) bb.putFloat(f);
        bb.rewind();
        return bb;
    }

    private static List<VectorFloat<?>> toVectorFloatList(float[][] data)
    {
        List<VectorFloat<?>> out = new ArrayList<>(data.length);
        for (float[] v : data) {
            out.add(VTS.createFloatVector(java.util.Arrays.copyOf(v, v.length)));
        }
        return out;
    }

    private void assertEquivalent(int count, int dim, VectorSimilarityFunction vsf, long seed)
            throws Exception
    {
        float[][] data = generate(seed, count, dim);

        // A: classic float[] build
        RandomAccessVectorValues ravvA = new ListRandomAccessVectorValues(toVectorFloatList(data), dim);
        var bspA = BuildScoreProvider.randomAccessScoreProvider(ravvA, vsf);
        GraphIndexBuilder builderA = new GraphIndexBuilder(bspA, dim, 16, 100, 1.2f, 1.2f, false);
        ImmutableGraphIndex graphA = TestUtil.buildSequentially(builderA, ravvA);

        // B: zero-copy ByteBuffer build
        ByteBuffer bb = toLittleEndian(data);
        RandomAccessVectorValues ravvB = new ByteBufferRandomAccessVectorValues(bb, count, dim);
        var bspB = BuildScoreProvider.randomAccessScoreProvider(ravvB, vsf);
        GraphIndexBuilder builderB = new GraphIndexBuilder(bspB, dim, 16, 100, 1.2f, 1.2f, false);
        ImmutableGraphIndex graphB = TestUtil.buildSequentially(builderB, ravvB);

        TestUtil.assertGraphEquals(graphA, graphB);

        // Query a handful of points against each and assert identical ordering and scores.
        Random qr = new Random(~seed);
        for (int q = 0; q < 10; q++) {
            int queryOrd = qr.nextInt(count);
            var queryFromList = ravvA.getVector(queryOrd);
            var queryFromBB = ravvB.getVector(queryOrd);

            var resA = GraphSearcher.search(queryFromList, 5, ravvA, vsf, graphA, Bits.ALL);
            var resB = GraphSearcher.search(queryFromBB, 5, ravvB, vsf, graphB, Bits.ALL);
            assertEquals("topK count differs", resA.getNodes().length, resB.getNodes().length);
            for (int i = 0; i < resA.getNodes().length; i++) {
                assertEquals("node ordering mismatch at i=" + i, resA.getNodes()[i].node, resB.getNodes()[i].node);
                assertEquals("score mismatch at i=" + i, resA.getNodes()[i].score, resB.getNodes()[i].score, 1e-5f);
            }
        }

        builderA.close();
        builderB.close();
    }

    @Test
    public void euclideanEquivalence() throws Exception
    {
        assertEquivalent(200, 64, VectorSimilarityFunction.EUCLIDEAN, 0xC001_C0DEL);
    }

    @Test
    public void dotProductEquivalence() throws Exception
    {
        assertEquivalent(150, 48, VectorSimilarityFunction.DOT_PRODUCT, 0xDEAD_BEEFL);
    }

    @Test
    public void cosineEquivalence() throws Exception
    {
        assertEquivalent(120, 32, VectorSimilarityFunction.COSINE, 0xABCDEF01L);
    }

    @Test
    public void largerDimensionEquivalence() throws Exception
    {
        assertEquivalent(64, 256, VectorSimilarityFunction.EUCLIDEAN, 0x5EED_5EEDL);
    }
}
