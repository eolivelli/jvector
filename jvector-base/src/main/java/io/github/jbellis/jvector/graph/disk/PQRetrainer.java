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

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Handles Product Quantization retraining for graph index compaction.
 * Performs balanced sampling across multiple source indexes and trains
 * a new PQ codebook optimized for the combined dataset.
 */
public class PQRetrainer {
    private static final Logger log = LoggerFactory.getLogger(PQRetrainer.class);
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int MIN_SAMPLES_PER_SOURCE = 1000;
    // Number of consecutive nodes to read per chunk before jumping to another location.
    // Keeping reads sequential within each chunk lets the OS read-ahead cover them,
    // avoiding the random I/O that would happen with per-node random sampling.
    private static final int SAMPLE_CHUNK_SIZE = 32;

    // Maximum number of source segments whose training vectors are extracted
    // concurrently. Each source is read by a single thread using its own View,
    // so this is the number of remote-storage reads that can be in flight at
    // once. Defaults to 16; overridable via the system property below for
    // deployments backed by slower or faster storage. Clamped to >= 1.
    private static final int IO_THREADS = Math.max(1,
            Integer.getInteger("jvector.pq.retrain.io.threads", 16));

    // Emit a progress log line every this many extracted training vectors, so
    // a slow-but-progressing extraction can be distinguished from a hang.
    private static final int PROGRESS_LOG_INTERVAL = 10_000;

    private final List<OnDiskGraphIndex> sources;
    private final List<FixedBitSet> liveNodes;
    private final List<Integer> numLiveNodesPerSource;
    private final int dimension;
    private final int numTotalNodes;
    /**
     * Optional factory that, given a source {@link OnDiskGraphIndex}, returns a
     * {@link ReaderSupplier} backed by a pre-downloaded or locally-buffered copy of
     * the index file. When non-null, {@link #extractVectorsSequential} uses it to
     * open Views via {@link OnDiskGraphIndex#getView(ReaderSupplier)} instead of the
     * default {@link OnDiskGraphIndex#getView()}, avoiding per-node block-cache reads
     * over remote storage (issue #599 Option B).
     *
     * <p>The factory may throw {@link RuntimeException} (including
     * {@link java.io.UncheckedIOException}) on failure; such exceptions surface as
     * the existing {@code RuntimeException("PQ retraining vector extraction failed")}
     * wrapper.
     */
    private final Function<OnDiskGraphIndex, ReaderSupplier> readerSupplierFactory;

    /**
     * Constructs a {@code PQRetrainer} without a custom reader-supplier factory.
     * Vector extraction uses the default {@link OnDiskGraphIndex#getView()}, which
     * routes reads through the block cache.
     */
    public PQRetrainer(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes, int dimension) {
        this(sources, liveNodes, dimension, null);
    }

    /**
     * Constructs a {@code PQRetrainer} with an optional reader-supplier factory.
     *
     * @param readerSupplierFactory when non-null, called once per source segment to
     *                              obtain a {@link ReaderSupplier} for bulk sequential
     *                              reads; the returned supplier is closed after all
     *                              vectors for that source are extracted
     */
    public PQRetrainer(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes, int dimension,
                       Function<OnDiskGraphIndex, ReaderSupplier> readerSupplierFactory) {
        this.sources = sources;
        this.liveNodes = liveNodes;
        this.dimension = dimension;
        this.readerSupplierFactory = readerSupplierFactory;

        this.numLiveNodesPerSource = new ArrayList<>(sources.size());
        int total = 0;
        for (int s = 0; s < sources.size(); s++) {
            int numLiveNodes = liveNodes.get(s).cardinality();
            total += numLiveNodes;
            this.numLiveNodesPerSource.add(numLiveNodes);
        }
        this.numTotalNodes = total;
    }

    /**
     * Trains a new Product Quantization codebook using balanced sampling across all source indexes.
     * All sampled vectors are read into memory up front, so ProductQuantization.compute() itself
     * performs no I/O.
     */
    public ProductQuantization retrain(VectorSimilarityFunction similarityFunction) {
        log.info("Training PQ using balanced sampling across sources");

        List<SampleRef> samples = sampleBalanced(ProductQuantization.MAX_PQ_TRAINING_SET_SIZE);

        // Sort by (source, node) so extractVectorsSequential reads each source's file
        // in ascending order, enabling read-ahead instead of random page faults.
        samples.sort(Comparator.comparingInt((SampleRef r) -> r.source).thenComparingInt(r -> r.node));

        log.info("Collected {} training samples", samples.size());

        // Extract vectors up front so ProductQuantization.compute() itself performs no
        // I/O. extractVectorsSequential reads each source ascending (read-ahead friendly)
        // and reads distinct sources concurrently so the per-read latency of remote
        // storage is hidden instead of serialized across thousands of samples.
        List<VectorFloat<?>> trainingVectors = extractVectorsSequential(samples);
        var ravv = new ListRandomAccessVectorValues(trainingVectors, dimension);

        FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
        ProductQuantization basePQ = fpq.getPQ();

        boolean center = similarityFunction == VectorSimilarityFunction.EUCLIDEAN;

        return ProductQuantization.compute(
                ravv,
                basePQ.getSubspaceCount(),
                basePQ.getClusterCount(),
                center
        );
    }

    /**
     * Performs balanced sampling across all source indexes to ensure proportional representation.
     * Guarantees minimum samples per source while respecting total sample budget.
     */
    private List<SampleRef> sampleBalanced(int totalSamples) {
        // If total live nodes <= totalSamples, return ALL
        if (numTotalNodes <= totalSamples) {
            List<SampleRef> all = new ArrayList<>(numTotalNodes);

            for (int s = 0; s < sources.size(); s++) {
                FixedBitSet live = liveNodes.get(s);

                for (int node = live.nextSetBit(0);
                     node != DocIdSetIterator.NO_MORE_DOCS;
                     node = live.nextSetBit(node + 1)) {
                    all.add(new SampleRef(s, node));
                }
            }

            return all;
        }

        final int MIN_PER_SOURCE = Math.min(MIN_SAMPLES_PER_SOURCE, totalSamples / sources.size());

        int[] quota = new int[sources.size()];
        int assigned = 0;

        // Proportional allocation
        for (int s = 0; s < sources.size(); s++) {
            quota[s] = Math.max(
                    MIN_PER_SOURCE,
                    (int) ((long) totalSamples * numLiveNodesPerSource.get(s) / numTotalNodes)
            );
            assigned += quota[s];
        }

        // Normalize down
        while (assigned > totalSamples) {
            for (int s = 0; s < sources.size() && assigned > totalSamples; s++) {
                if (quota[s] > MIN_PER_SOURCE) {
                    quota[s]--;
                    assigned--;
                }
            }
        }

        // Normalize up
        while (assigned < totalSamples) {
            for (int s = 0; s < sources.size() && assigned < totalSamples; s++) {
                quota[s]++;
                assigned++;
            }
        }

        List<SampleRef> samples = new ArrayList<>(totalSamples);
        ThreadLocalRandom rand = ThreadLocalRandom.current();

        for (int s = 0; s < sources.size(); s++) {
            FixedBitSet live = liveNodes.get(s);
            int max = live.length();
            int numChunks = (max + SAMPLE_CHUNK_SIZE - 1) / SAMPLE_CHUNK_SIZE;

            // Build a shuffled chunk order so samples are representative but
            // each chunk is read sequentially to minimize page faults.
            // Fisher-Yates shuffle
            int[] chunkOrder = new int[numChunks];
            for (int i = 0; i < numChunks; i++) chunkOrder[i] = i;
            for (int i = numChunks - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                int tmp = chunkOrder[i];
                chunkOrder[i] = chunkOrder[j];
                chunkOrder[j] = tmp;
            }

            int count = 0;
            outer:
            for (int ci = 0; ci < numChunks; ci++) {
                int start = chunkOrder[ci] * SAMPLE_CHUNK_SIZE;
                int end = Math.min(max, start + SAMPLE_CHUNK_SIZE);
                for (int node = start; node < end; node++) {
                    if (live.get(node)) {
                        samples.add(new SampleRef(s, node));
                        if (++count >= quota[s]) break outer;
                    }
                }
            }
        }

        return samples;
    }

    /**
     * Reads the sampled vectors into memory.
     *
     * <p>The caller pre-sorts {@code samples} by (source, node) so each source's
     * sub-list is contiguous and ascending. Sources are then extracted
     * <em>concurrently</em>: each source is processed by a single worker thread
     * using its own {@link OnDiskGraphIndex.View} (and therefore its own
     * {@code RandomAccessReader}), so concurrent reads are safe and up to
     * {@link #IO_THREADS} remote-storage reads can be in flight at once. Within
     * a source the reads stay ascending, preserving read-ahead friendliness.
     *
     * <p>This matters for graphs backed by remote storage (e.g. an object
     * store): there each {@code getVectorInto} is a network round-trip with no
     * OS read-ahead, so a single-threaded loop over thousands of samples
     * serializes thousands of round-trips. Reading sources in parallel hides
     * that latency. The returned list order is unspecified — irrelevant, since
     * it only feeds {@link ProductQuantization#compute} as an unordered
     * training set.
     */
    private List<VectorFloat<?>> extractVectorsSequential(List<SampleRef> samples) {
        if (samples.isEmpty()) {
            return new ArrayList<>();
        }

        // Group the pre-sorted samples by source. The caller sorted by
        // (source, node) so each source's sub-list is contiguous and ascending.
        Map<Integer, List<SampleRef>> bySource = new LinkedHashMap<>();
        for (SampleRef ref : samples) {
            bySource.computeIfAbsent(ref.source, k -> new ArrayList<>()).add(ref);
        }

        int parallelism = Math.min(bySource.size(), IO_THREADS);
        // Order is irrelevant for PQ codebook training, so a synchronized list
        // collecting from all workers is sufficient.
        List<VectorFloat<?>> vectors =
                Collections.synchronizedList(new ArrayList<>(samples.size()));
        AtomicInteger progress = new AtomicInteger();
        int totalSamples = samples.size();
        long startNanos = System.nanoTime();

        ExecutorService pool = Executors.newFixedThreadPool(parallelism, r -> {
            Thread t = new Thread(r, "pq-retrain-io");
            t.setDaemon(true);
            return t;
        });
        try {
            List<Future<?>> futures = new ArrayList<>(bySource.size());
            for (Map.Entry<Integer, List<SampleRef>> entry : bySource.entrySet()) {
                final int source = entry.getKey();
                final List<SampleRef> group = entry.getValue();
                futures.add(pool.submit(() -> {
                    // One View — and one RandomAccessReader — per task. Never
                    // shared across threads, so concurrent extraction is safe.
                    //
                    // When a readerSupplierFactory is present, use it to open a View
                    // backed by a pre-downloaded or locally-buffered copy of the index
                    // file instead of the default block-cache reader. This eliminates
                    // per-node remote-storage round-trips (issue #599 Option B).
                    final OnDiskGraphIndex odg = sources.get(source);
                    final ReaderSupplier supplierForSource;
                    final OnDiskGraphIndex.View view;
                    if (readerSupplierFactory != null) {
                        supplierForSource = readerSupplierFactory.apply(odg);
                        try {
                            view = odg.getView(supplierForSource);
                        } catch (IOException e) {
                            try {
                                supplierForSource.close();
                            } catch (IOException suppressed) {
                                e.addSuppressed(suppressed);
                            }
                            throw new java.io.UncheckedIOException(e);
                        }
                    } else {
                        supplierForSource = null;
                        view = (OnDiskGraphIndex.View) odg.getView();
                    }
                    try {
                        VectorFloat<?> scratch = vectorTypeSupport.createFloatVector(dimension);
                        for (SampleRef ref : group) {
                            view.getVectorInto(ref.node, scratch, 0);
                            vectors.add(scratch.copy());
                            int done = progress.incrementAndGet();
                            if (done % PROGRESS_LOG_INTERVAL == 0) {
                                log.info("PQ retraining: extracted {}/{} training vectors",
                                        done, totalSamples);
                            }
                        }
                    } finally {
                        try {
                            view.close();
                        } catch (IOException ioe) {
                            log.warn("Failed to close source {} view during PQ retraining",
                                    source, ioe);
                        }
                        if (supplierForSource != null) {
                            try {
                                supplierForSource.close();
                            } catch (IOException ioe) {
                                log.warn("Failed to close bulk reader supplier for source {} during PQ retraining",
                                        source, ioe);
                            }
                        }
                    }
                }));
            }
            // Wait for completion; surface the first failure.
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("PQ retraining vector extraction interrupted", ie);
                } catch (ExecutionException ee) {
                    throw new RuntimeException("PQ retraining vector extraction failed",
                            ee.getCause());
                }
            }
        } finally {
            pool.shutdownNow();
        }

        log.info("PQ retraining: extracted {} training vectors from {} sources in {} ms ({} threads)",
                vectors.size(), bySource.size(),
                (System.nanoTime() - startNanos) / 1_000_000L, parallelism);
        return vectors;
    }

    /**
     * Reference to a sampled vector from a specific source index.
     */
    private static final class SampleRef {
        final int source;
        final int node;

        SampleRef(int source, int node) {
            this.source = source;
            this.node = node;
        }
    }

}
