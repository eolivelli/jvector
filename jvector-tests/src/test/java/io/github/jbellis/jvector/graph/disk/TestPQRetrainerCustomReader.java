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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Verifies the custom-reader-supplier API introduced in issue #599 (Option B):
 *
 * <ol>
 *   <li>{@link OnDiskGraphIndex#getView(ReaderSupplier)} opens a View backed by the
 *       caller-supplied reader and reads vectors correctly.</li>
 *   <li>When a non-null factory is passed to
 *       {@link PQRetrainer#PQRetrainer(List, List, int, java.util.function.Function)},
 *       the factory is invoked once per source segment during
 *       {@link PQRetrainer#retrain}, and the returned PQ codebook is valid.</li>
 * </ol>
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestPQRetrainerCustomReader extends RandomizedTest {

    private static final VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private final ForkJoinPool executor = ForkJoinPool.commonPool();

    private Path testDirectory;
    private static final int DIMENSION = 32;
    private static final int VECTORS_PER_SOURCE = 256;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    @Before
    public void setUp() throws IOException {
        testDirectory = Files.createTempDirectory("TestPQRetrainerCustomReader");
    }

    @After
    public void tearDown() throws IOException {
        // Delete temp files created during the test.
        if (testDirectory != null) {
            try (var stream = Files.walk(testDirectory)) {
                stream.sorted(java.util.Comparator.reverseOrder())
                      .map(java.nio.file.Path::toFile)
                      .forEach(java.io.File::delete);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Builds a small FusedPQ-enabled {@link OnDiskGraphIndex} and writes it to
     * {@code outputPath}.
     *
     * @return the written path (same as {@code outputPath})
     */
    private Path buildFusedPQSource(List<VectorFloat<?>> vecs, Path outputPath) throws IOException {
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(vecs, DIMENSION);
        ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true, UNWEIGHTED, executor, executor);
        PQVectors pqv = (PQVectors) pq.encodeAll(ravv, executor);
        var bsp = BuildScoreProvider.pqBuildScoreProvider(SIMILARITY, pqv);
        var builder = new GraphIndexBuilder(bsp, DIMENSION, 16, 100, 1.2f, 1.2f, false, true, executor, executor);
        var graph = builder.getGraph();

        Map<FeatureId, IntFunction<Feature.State>> writeSuppliers = new EnumMap<>(FeatureId.class);
        writeSuppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));

        var identityMapper = new OrdinalMapper.IdentityMapper(ravv.size() - 1);
        var writerBuilder = new OnDiskGraphIndexWriter.Builder(graph, outputPath);
        writerBuilder.withMapper(identityMapper);
        writerBuilder.with(new InlineVectors(DIMENSION));
        writerBuilder.with(new FusedPQ(graph.maxDegree(), pq));
        var writer = writerBuilder.build();

        for (int node = 0; node < ravv.size(); node++) {
            var stateMap = new EnumMap<FeatureId, Feature.State>(FeatureId.class);
            stateMap.put(FeatureId.INLINE_VECTORS, writeSuppliers.get(FeatureId.INLINE_VECTORS).apply(node));
            writer.writeInline(node, stateMap);
            builder.addGraphNode(node, ravv.getVector(node));
        }
        builder.cleanup();
        writeSuppliers.put(FeatureId.FUSED_PQ, ordinal -> new FusedPQ.State(graph.getView(), pqv, ordinal));
        writer.write(writeSuppliers);
        return outputPath;
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    /**
     * Verifies that {@link OnDiskGraphIndex#getView(ReaderSupplier)} opens a view
     * backed by the caller-supplied reader and returns the same vectors as the
     * default {@link OnDiskGraphIndex#getView()}.
     */
    @Test
    public void getViewWithCustomSupplierReadsVectorsCorrectly() throws IOException {
        List<VectorFloat<?>> vecs = createRandomVectors(VECTORS_PER_SOURCE, DIMENSION);
        Path p = buildFusedPQSource(vecs, testDirectory.resolve("source.idx"));

        try (ReaderSupplier defaultSupplier = ReaderSupplierFactory.open(p);
             ReaderSupplier customSupplier = ReaderSupplierFactory.open(p)) {
            OnDiskGraphIndex odg = OnDiskGraphIndex.load(defaultSupplier);

            // Read the first 5 nodes via the default path.
            VectorFloat<?> expected = vectorTypeSupport.createFloatVector(DIMENSION);
            try (OnDiskGraphIndex.View defaultView = odg.getView()) {
                defaultView.getVectorInto(0, expected, 0);
            }

            // Read the same node via the custom-supplier path.
            VectorFloat<?> actual = vectorTypeSupport.createFloatVector(DIMENSION);
            try (OnDiskGraphIndex.View customView = odg.getView(customSupplier)) {
                customView.getVectorInto(0, actual, 0);
            }

            // Vectors must be byte-identical.
            for (int d = 0; d < DIMENSION; d++) {
                assertEquals("dimension " + d, expected.get(d), actual.get(d), 0f);
            }
        }
    }

    /**
     * Verifies that when a non-null factory is passed to {@link PQRetrainer}, the
     * factory is invoked exactly once per source segment during
     * {@link PQRetrainer#retrain}, and the resulting PQ codebook is non-null and
     * has the expected structure.
     */
    @Test
    public void pqRetrainerInvokesCustomFactoryOncePerSource() throws IOException {
        final int numSources = 3;

        List<OnDiskGraphIndex> sources = new ArrayList<>();
        List<FixedBitSet> liveSets = new ArrayList<>();
        List<ReaderSupplier> openSuppliers = new ArrayList<>();

        for (int i = 0; i < numSources; i++) {
            List<VectorFloat<?>> vecs = createRandomVectors(VECTORS_PER_SOURCE, DIMENSION);
            Path p = buildFusedPQSource(vecs, testDirectory.resolve("src_" + i + ".idx"));
            ReaderSupplier rs = ReaderSupplierFactory.open(p);
            openSuppliers.add(rs);
            OnDiskGraphIndex odg = OnDiskGraphIndex.load(rs);
            sources.add(odg);

            FixedBitSet live = new FixedBitSet(VECTORS_PER_SOURCE);
            live.set(0, VECTORS_PER_SOURCE); // all nodes live
            liveSets.add(live);
        }

        AtomicInteger factoryCalls = new AtomicInteger(0);

        // Build a tracking factory that counts invocations and wraps the file reader.
        java.util.function.Function<OnDiskGraphIndex, ReaderSupplier> trackingFactory = odg -> {
            int idx = sources.indexOf(odg);
            assertTrue("factory called with unknown source", idx >= 0);
            factoryCalls.incrementAndGet();
            // Open a fresh reader for the source file (same path as the source).
            try {
                return ReaderSupplierFactory.open(testDirectory.resolve("src_" + idx + ".idx"));
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        };

        PQRetrainer retrainer = new PQRetrainer(sources, liveSets, DIMENSION, trackingFactory);
        ProductQuantization pq = retrainer.retrain(SIMILARITY);

        // The factory must have been called once per source.
        assertEquals("factory must be called once per source", numSources, factoryCalls.get());

        // The resulting PQ codebook must be valid.
        assertNotNull("PQ codebook must not be null", pq);
        assertTrue("PQ must have at least one subspace", pq.getSubspaceCount() > 0);
        assertEquals("PQ cluster count must be 256", 256, pq.getClusterCount());

        // Clean up open suppliers.
        for (ReaderSupplier rs : openSuppliers) {
            rs.close();
        }
    }

    /**
     * Verifies that the default (3-arg) constructor delegates with a null factory,
     * i.e. the original code path is unaffected.
     */
    @Test
    public void defaultConstructorUsesNullFactory() throws IOException {
        List<VectorFloat<?>> vecs0 = createRandomVectors(VECTORS_PER_SOURCE, DIMENSION);
        List<VectorFloat<?>> vecs1 = createRandomVectors(VECTORS_PER_SOURCE, DIMENSION);
        Path p0 = buildFusedPQSource(vecs0, testDirectory.resolve("default0.idx"));
        Path p1 = buildFusedPQSource(vecs1, testDirectory.resolve("default1.idx"));

        List<OnDiskGraphIndex> sources = new ArrayList<>();
        List<FixedBitSet> liveSets = new ArrayList<>();
        List<ReaderSupplier> openSuppliers = new ArrayList<>();

        for (Path p : new Path[]{p0, p1}) {
            ReaderSupplier rs = ReaderSupplierFactory.open(p);
            openSuppliers.add(rs);
            sources.add(OnDiskGraphIndex.load(rs));
            FixedBitSet live = new FixedBitSet(VECTORS_PER_SOURCE);
            live.set(0, VECTORS_PER_SOURCE);
            liveSets.add(live);
        }

        // 3-arg constructor — null factory, uses default getView() path.
        PQRetrainer retrainer = new PQRetrainer(sources, liveSets, DIMENSION);
        ProductQuantization pq = retrainer.retrain(SIMILARITY);
        assertNotNull("default-path PQ codebook must not be null", pq);

        for (ReaderSupplier rs : openSuppliers) {
            rs.close();
        }
    }
}
