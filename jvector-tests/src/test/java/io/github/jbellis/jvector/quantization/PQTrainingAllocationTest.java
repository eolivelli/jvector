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
package io.github.jbellis.jvector.quantization;

import com.sun.management.ThreadMXBean;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.ArraySliceVectorFloat;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Verifies that PQ codebook training does not allocate a fresh {@code float[]} per
 * (training-vector × subspace). The subvector extraction path now returns zero-copy views
 * via {@link ArraySliceVectorFloat}, so the per-subvector allocation that showed up in
 * async-profiler flamegraphs during HerdDB indexing should be gone.
 *
 * <p>Uses {@code ThreadMXBean.getThreadAllocatedBytes} to measure; the assertion is relative
 * to the number of training vectors (not an absolute byte budget) so the test is stable
 * across JDK versions and SIMD backends.
 */
public class PQTrainingAllocationTest
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void getSubVectorReturnsViewNotCopy()
    {
        int dim = 128;
        VectorFloat<?> v = VTS.createFloatVector(dim);
        for (int i = 0; i < dim; i++) v.set(i, i * 0.5f);

        int[][] ranges = {{16, 0}, {16, 16}, {16, 32}};
        VectorFloat<?> sub0 = ProductQuantization.getSubVector(v, 0, ranges);
        VectorFloat<?> sub1 = ProductQuantization.getSubVector(v, 1, ranges);
        assertEquals(16, sub0.length());
        assertEquals(16, sub1.length());

        // Mutating the source must reflect through the views — proves no copy.
        v.set(5, 999f);  // index 5 is in sub0 (offset 0..15)
        assertEquals(999f, sub0.get(5), 0f);

        v.set(20, 888f); // index 20 is in sub1 (offset 16..31)
        assertEquals(888f, sub1.get(4), 0f);
    }

    @Test
    public void pqTrainingAllocationStaysBoundedPerVector() throws Exception
    {
        int dim = 64;
        int n = 2_000;
        int m = 8;
        int clusterCount = 16;

        Random r = new Random(0xC0DEFACE);
        List<VectorFloat<?>> training = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dim);
            for (int j = 0; j < dim; j++) v.set(j, r.nextFloat() * 2f - 1f);
            training.add(v);
        }
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(training, dim);

        // Warm-up — trigger class loading, JIT
        ProductQuantization.compute(ravv, m, clusterCount, false);

        ThreadMXBean tmx = (ThreadMXBean) ManagementFactory.getThreadMXBean();
        long tid = Thread.currentThread().getId();
        long before = tmx.getThreadAllocatedBytes(tid);
        ProductQuantization pq = ProductQuantization.compute(ravv, m, clusterCount, false);
        long after = tmx.getThreadAllocatedBytes(tid);

        long perTrainingVector = (after - before) / n;
        System.out.printf("PQTrainingAllocationTest: %d B allocated per training vector (n=%d, dim=%d, M=%d, K=%d)%n",
                perTrainingVector, n, dim, m, clusterCount);

        // If getSubVector materialized every subvector, we'd allocate M*(dim/M)*4 = dim*4
        // = 256 B per training vector just for subvectors, plus wrapper overhead. With
        // view-based extraction the per-vector cost is dominated by KMeans centroids, which
        // amortize over the whole training set. Guard at 2x the pre-fix baseline minus
        // subvectors; if someone reintroduces the allocation, the ratio explodes.
        long budget = (long) dim * Float.BYTES * 2;
        assertTrue("per-vector allocation " + perTrainingVector + " exceeds budget " + budget
                        + " — getSubVector may have regressed to materialization",
                perTrainingVector < budget);
        assertEquals(m, pq.getSubspaceCount());
    }
}
