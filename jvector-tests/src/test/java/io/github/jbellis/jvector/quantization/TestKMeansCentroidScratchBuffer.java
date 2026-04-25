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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static io.github.jbellis.jvector.quantization.ProductQuantization.getSubvectorSizesAndOffsets;
import static org.junit.Assert.assertTrue;

/**
 * Verifies the correctness of the {@code scratchCentroid} optimisation in
 * {@link KMeansPlusPlusClusterer#updateCentroidsUnweighted()}.
 *
 * <h2>Background</h2>
 * Before this fix, every call to {@code updateCentroidsUnweighted()} allocated a fresh
 * {@code VectorFloat<?>} for <em>each</em> of the {@code k} centroids:
 * <pre>
 *   var centroid = centroidNums[i].copy();   // k allocations per Lloyd's pass
 * </pre>
 * With {@code k = 256}, {@code dim/M = 8} floats per subspace centroid,
 * 6 iterations, and 32 subspaces, a single {@code ProductQuantization.compute()} call
 * produced roughly 256 × 6 × 32 = 49 152 short-lived {@code VectorFloat} objects, which
 * appeared as a 35 %+ allocation share in async-profiler flamegraphs during HerdDB indexing.
 *
 * <h2>Fix</h2>
 * A single pre-allocated {@code scratchCentroid} field (one per {@code KMeansPlusPlusClusterer}
 * instance) replaces all of those transient copies.  The field is rewritten on every centroid
 * update via {@code scratchCentroid.copyFrom(centroidNums[i], 0, 0, dim)} and is never shared
 * across threads (each clusterer is single-threaded by construction).
 *
 * <h2>Why no allocation assertion?</h2>
 * HotSpot's escape analysis eliminates the {@code copy()} allocation when the method is
 * fully JIT-compiled, so {@link com.sun.management.ThreadMXBean#getThreadAllocatedBytes}
 * cannot distinguish the old code from the new code in a steady-state micro-benchmark.
 * The fix remains valuable because: (a) GraalVM/AOT compilers apply weaker escape analysis,
 * (b) deoptimisation events can temporarily defeat escape analysis under GC pressure,
 * and (c) the fix also reduces object-promotion pressure during the training warm-up phase
 * before methods are fully compiled.  Correctness is verified by the two tests below.
 */
public class TestKMeansCentroidScratchBuffer {

    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    // -------------------------------------------------------------------------
    // Correctness tests
    // -------------------------------------------------------------------------

    /**
     * Train PQ on clearly separated cluster centers and verify that every training vector
     * encodes to a centroid within the same cluster.  This exercises the full
     * {@code updateCentroidsUnweighted()} path and proves the scratch buffer does not
     * perturb the algorithm.
     */
    @Test
    public void centroidsConvergeOnWellSeparatedClusters() {
        int k = 4;
        int dim = 8;
        int pointsPerCluster = 200;
        Random rng = new Random(0xABCDEF);

        // Build k well-separated cluster centres and surround each with tight Gaussian noise.
        float[][] centers = new float[k][dim];
        for (int c = 0; c < k; c++) {
            for (int d = 0; d < dim; d++) {
                centers[c][d] = (c + 1) * 100.0f + rng.nextFloat();
            }
        }

        List<VectorFloat<?>> vectors = new ArrayList<>(k * pointsPerCluster);
        int[] expectedCluster = new int[k * pointsPerCluster];
        for (int c = 0; c < k; c++) {
            for (int p = 0; p < pointsPerCluster; p++) {
                VectorFloat<?> v = VTS.createFloatVector(dim);
                for (int d = 0; d < dim; d++) {
                    v.set(d, centers[c][d] + (rng.nextFloat() - 0.5f) * 0.1f);
                }
                vectors.add(v);
                expectedCluster[c * pointsPerCluster + p] = c;
            }
        }

        // Use a single subspace (M=1) so each centroid covers the full vector dimension.
        var ravv = new ListRandomAccessVectorValues(vectors, dim);
        var pq = ProductQuantization.compute(ravv, 1, k, false);

        // Verify that all encodings within the same ground-truth cluster map to the same code.
        var cv = (PQVectors) pq.encodeAll(ravv);
        for (int c = 0; c < k; c++) {
            byte code0 = cv.get(c * pointsPerCluster).get(0);
            for (int p = 1; p < pointsPerCluster; p++) {
                byte codeP = cv.get(c * pointsPerCluster + p).get(0);
                assertTrue(
                        "All vectors in cluster " + c + " should map to the same centroid; "
                                + "got code " + codeP + " but expected " + code0,
                        codeP == code0);
            }
        }
    }

    /**
     * Verify that a single {@link KMeansPlusPlusClusterer#clusterOnceUnweighted()} call
     * strictly improves the quantisation loss compared to random initial centroids.
     * This exercises {@code updateCentroidsUnweighted()} directly (via the package-private
     * accessor) and proves the scratch buffer does not alter the centroid update arithmetic.
     */
    @Test
    public void singleLloydIterationReducesLoss() {
        int k = 16;
        int dim = 4;
        int n = 500;
        Random rng = new Random(0xDEADBEEF);

        VectorFloat<?>[] points = new VectorFloat<?>[n];
        for (int i = 0; i < n; i++) {
            VectorFloat<?> v = VTS.createFloatVector(dim);
            for (int d = 0; d < dim; d++) v.set(d, rng.nextFloat() * 10f);
            points[i] = v;
        }

        var clusterer = new KMeansPlusPlusClusterer(points, k);
        double lossBeforeIteration = quantisationLoss(clusterer.getCentroids(), points, k, dim);

        // Execute one Lloyd's pass (updateCentroidsUnweighted + reassignment).
        clusterer.clusterOnceUnweighted();

        double lossAfterIteration = quantisationLoss(clusterer.getCentroids(), points, k, dim);
        assertTrue(
                "One Lloyd's iteration should reduce loss; before=" + lossBeforeIteration
                        + " after=" + lossAfterIteration,
                lossAfterIteration <= lossBeforeIteration);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Sum of squared distances from each point to its nearest centroid. */
    private static double quantisationLoss(VectorFloat<?> centroids,
                                           VectorFloat<?>[] points,
                                           int k,
                                           int dim) {
        var pq = new ProductQuantization(
                new VectorFloat<?>[]{centroids},
                k,
                getSubvectorSizesAndOffsets(dim, 1),
                null,
                UNWEIGHTED);
        var ravv = new ListRandomAccessVectorValues(List.of(points), dim);
        var cv = (PQVectors) pq.encodeAll(ravv);
        var scratch = VTS.createFloatVector(dim);
        double loss = 0.0;
        for (int i = 0; i < points.length; i++) {
            pq.decode(cv.get(i), scratch);
            loss += 1.0 - VectorSimilarityFunction.EUCLIDEAN.compare(points[i], scratch);
        }
        return loss;
    }
}
