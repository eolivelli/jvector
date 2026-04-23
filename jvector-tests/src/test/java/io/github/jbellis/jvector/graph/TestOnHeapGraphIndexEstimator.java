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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Validates {@link OnHeapGraphIndex#estimatedBytesPerNode(int, float)} — the static helper
 * external services use to predict graph overhead before building. The estimator must agree
 * with the formula the running graph uses ({@link OnHeapGraphIndex#ramBytesUsedOneNode(int)}).
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnHeapGraphIndexEstimator extends RandomizedTest {

    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void testEstimatorMatchesInstanceMethod() throws IOException {
        int M = 16;
        float overflowRatio = 1.2f;
        OnHeapGraphIndex graph = buildSmallGraph(M, overflowRatio, /* nodes */ 200, /* dim */ 16, /* hierarchy */ true);
        try {
            long estimate = OnHeapGraphIndex.estimatedBytesPerNode(M, overflowRatio);
            long actual = graph.ramBytesUsedOneNode(0);
            Assert.assertEquals("static estimator must equal instance per-node cost for level 0",
                    actual, estimate);
        } finally {
            graph.close();
        }
    }

    @Test
    public void testEstimatorScalesAcrossDegrees() {
        // Higher M → higher per-node cost, monotonically.
        long prev = 0;
        for (int m : new int[]{4, 8, 16, 32, 64}) {
            long est = OnHeapGraphIndex.estimatedBytesPerNode(m, 1.2f);
            Assert.assertTrue("per-node cost must grow with M (m=" + m + ", est=" + est + ", prev=" + prev + ")",
                    est > prev);
            prev = est;
        }
    }

    @Test
    public void testEstimatorScalesWithOverflow() {
        long lo = OnHeapGraphIndex.estimatedBytesPerNode(16, 1.0f);
        long mid = OnHeapGraphIndex.estimatedBytesPerNode(16, 1.5f);
        long hi = OnHeapGraphIndex.estimatedBytesPerNode(16, 2.0f);
        Assert.assertTrue(lo < mid);
        Assert.assertTrue(mid < hi);
    }

    @Test
    public void testEstimatorMatchesActualPerNodeAtScale() throws IOException {
        // Build a small graph with a hierarchy. The dense base layer's
        // ramBytesUsedOneLayer(0) / size(0) (after subtracting fixed-cost per-layer
        // overhead) must agree with the static estimator within ~5% — they share the
        // same Neighbors.ramBytesUsed formula.
        int M = 16;
        float overflowRatio = 1.2f;
        OnHeapGraphIndex graph = buildSmallGraph(M, overflowRatio, 250, 16, true);
        try {
            long estimate = OnHeapGraphIndex.estimatedBytesPerNode(M, overflowRatio);
            int sz = graph.size(0);
            Assert.assertTrue("graph too small to evaluate", sz >= 50);
            // ramBytesUsedOneNode is a public instance method — it returns exactly the
            // same per-node figure as the static estimator does for the same parameters.
            long actualPerNode = graph.ramBytesUsedOneNode(0);
            double ratio = (double) estimate / actualPerNode;
            Assert.assertTrue("estimate=" + estimate + " actual=" + actualPerNode
                            + " ratio=" + ratio, ratio > 0.95 && ratio < 1.05);
        } finally {
            graph.close();
        }
    }

    private OnHeapGraphIndex buildSmallGraph(int M, float overflowRatio, int n, int dim, boolean addHierarchy) throws IOException {
        ArrayList<VectorFloat<?>> vectors = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dim);
            for (int d = 0; d < dim; d++) {
                v.set(d, (float) Math.random());
            }
            vectors.add(v);
        }
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vectors, dim);
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(
                ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(
                bsp, dim, M, /* beamWidth */ 100, overflowRatio, /* alpha */ 1.2f, addHierarchy)) {
            return (OnHeapGraphIndex) builder.build(ravv);
        }
    }
}
