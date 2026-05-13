/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Validates the async-IO pipeline added on the layer-0 FusedPQ search path.
 * <p>
 * Two things to check:
 * <ol>
 *   <li>The default {@link RandomAccessReader#readRangeAsync(long, int)} fallback returns the same
 *       bytes as a {@code seek}+{@code readFully} pair and does not change the reader's position.</li>
 *   <li>With pipeline enabled, search returns the same node ids and scores as the sync path on a
 *       deterministic FusedPQ graph (bit-equivalence — IO scheduling change must not affect the
 *       traversal).</li>
 * </ol>
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestAsyncPipelineSearch extends RandomizedTest {

    private Path testDirectory;
    private Random random;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        random = getRandom();
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testDefaultReadRangeAsyncMatchesSync() throws IOException {
        Path file = testDirectory.resolve("bytes.bin");
        byte[] expected = new byte[1024];
        random.nextBytes(expected);
        Files.write(file, expected);

        try (var supplier = new SimpleMappedReader.Supplier(file);
             var reader = supplier.get()) {
            reader.seek(7);
            // capture position before
            long beforePos = reader.getPosition();
            ByteBuffer got = reader.readRangeAsync(100, 200).join();
            // position must be unchanged
            assertEquals(beforePos, reader.getPosition());
            // bytes must match
            byte[] actual = new byte[200];
            got.get(actual);
            assertArrayEquals(Arrays.copyOfRange(expected, 100, 300), actual);
        }
    }

    @Test
    public void testAsyncPipelineMatchesSyncOnFusedPQ() throws IOException {
        int size = 500;
        int dim = 32;
        var vectors = MockVectorValues.fromValues(createRandomFloatVectors(size, dim, random));

        var simFn = VectorSimilarityFunction.EUCLIDEAN;
        int topK = 10;
        int rerankK = 40;

        var builder = new GraphIndexBuilder(vectors, simFn, 32, 32, 1.2f, 1.2f, false);
        var tempGraph = builder.build(vectors);
        var pq = ProductQuantization.compute(vectors, 8, 256, false);
        var pqv = (PQVectors) pq.encodeAll(vectors);

        var outputPath = testDirectory.resolve("graph_fpq");
        TestUtil.writeFusedGraph(tempGraph, vectors, pqv, FeatureId.INLINE_VECTORS, outputPath);

        try (var readerSupplier = new SimpleMappedReader.Supplier(outputPath);
             var graph = OnDiskGraphIndex.load(readerSupplier, 0)) {

            for (int q = 0; q < 25; q++) {
                VectorFloat<?> query = TestUtil.randomVector(random, dim);

                // sync baseline
                int[] syncNodes;
                float[] syncScores;
                try (var searcher = new GraphSearcher(graph)) {
                    var ssp = fusedScoreProvider(searcher.getView(), query, simFn);
                    var r = searcher.search(ssp, topK, rerankK, 0f, 0f, Bits.ALL);
                    syncNodes = nodeIds(r);
                    syncScores = nodeScores(r);
                }

                // async pipeline
                int[] asyncNodes;
                float[] asyncScores;
                try (var searcher = new GraphSearcher(graph)) {
                    searcher.setAsyncPipelineEnabled(true);
                    var ssp = fusedScoreProvider(searcher.getView(), query, simFn);
                    var r = searcher.search(ssp, topK, rerankK, 0f, 0f, Bits.ALL);
                    asyncNodes = nodeIds(r);
                    asyncScores = nodeScores(r);
                }

                // bit-equivalent: same node ids in the same order, same scores.
                assertArrayEquals("query " + q + " nodes differ", syncNodes, asyncNodes);
                for (int i = 0; i < syncScores.length; i++) {
                    assertEquals("query " + q + " score[" + i + "]",
                            syncScores[i], asyncScores[i], 1e-6f);
                }
            }
        }
    }

    private static SearchScoreProvider fusedScoreProvider(ImmutableGraphIndex.View view,
                                                          VectorFloat<?> query,
                                                          VectorSimilarityFunction simFn) {
        var scoringView = (ImmutableGraphIndex.ScoringView) view;
        ScoreFunction.ApproximateScoreFunction asf =
                scoringView.approximateScoreFunctionFor(query, simFn);
        var rr = scoringView.rerankerFor(query, simFn);
        return new DefaultSearchScoreProvider(asf, rr);
    }

    private static int[] nodeIds(SearchResult r) {
        return Arrays.stream(r.getNodes()).mapToInt(ns -> ns.node).toArray();
    }

    private static float[] nodeScores(SearchResult r) {
        SearchResult.NodeScore[] ns = r.getNodes();
        float[] out = new float[ns.length];
        for (int i = 0; i < ns.length; i++) out[i] = ns[i].score;
        return out;
    }
}
