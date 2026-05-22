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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;

import java.nio.file.Path;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;
import java.util.stream.IntStream;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.*;

public final class OnDiskGraphIndexCompactor implements Accountable {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final Logger log = LoggerFactory.getLogger(OnDiskGraphIndexCompactor.class);

    // Compaction constants
    private static final float DIVERSITY_ALPHA_STEP = 0.2f;
    private static final int BEAM_WIDTH_MULTIPLIER = 2;
    private static final int TARGET_BATCHES_PER_SOURCE = 40;
    private static final int TARGET_NODES_PER_BATCH = 128;
    private static final int MIN_SEARCH_TOP_K = 2;
    private static final int SEARCH_TOP_K_MULTIPLIER = 2;

    private final List<OnDiskGraphIndex> sources;
    private final List<FixedBitSet> liveNodes;
    private final List<Integer> numLiveNodesPerSource;
    private final List<OrdinalMapper> remappers;
    private final List<Integer> maxDegrees;

    private final int dimension;
    private int maxOrdinal = -1;
    private int numTotalNodes = 0;
    private boolean ownsExecutor = false;
    private final ForkJoinPool executor;
    private final int taskWindowSize;
    private final VectorSimilarityFunction similarityFunction;
    /**
     * Optional factory for bulk sequential readers used during PQ retraining.
     * When non-null, {@link #resolvePQFromSources} passes it to {@link PQRetrainer}
     * so that training-vector extraction avoids per-node block-cache round-trips
     * (issue #599 Option B). {@code null} uses the default block-cache path.
     */
    private final Function<OnDiskGraphIndex, ReaderSupplier> readerSupplierFactory;

    /**
     * Constructs a new OnDiskGraphIndexCompactor to merge multiple graph indexes.
     * Uses the default block-cache path for PQ retraining vector extraction.
     *
     * @see #OnDiskGraphIndexCompactor(List, List, List, VectorSimilarityFunction, ForkJoinPool, Function)
     */
    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor) {
        this(sources, liveNodes, remappers, similarityFunction, executor, null);
    }

    /**
     * Constructs a new OnDiskGraphIndexCompactor to merge multiple graph indexes,
     * with an optional reader-supplier factory for bulk PQ retraining I/O.
     *
     * @param readerSupplierFactory when non-null, called once per source segment
     *                              during PQ retraining to obtain a
     *                              {@link ReaderSupplier} backed by a pre-downloaded
     *                              or locally-buffered copy of the index file; avoids
     *                              per-node block-cache round-trips (issue #599
     *                              Option B); pass {@code null} for the default path
     */
    public OnDiskGraphIndexCompactor(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers,
            VectorSimilarityFunction similarityFunction,
            ForkJoinPool executor,
            Function<OnDiskGraphIndex, ReaderSupplier> readerSupplierFactory) {
        checkBeforeCompact(sources, liveNodes, remappers);

        int threads = Runtime.getRuntime().availableProcessors();
        if (executor != null) {
            this.executor = executor;
        } else {
            this.executor = new ForkJoinPool(threads);
            this.ownsExecutor = true;
        }
        this.taskWindowSize = threads;

        this.sources = sources;
        this.remappers = remappers;
        this.liveNodes = liveNodes;
        this.numLiveNodesPerSource = new ArrayList<>(this.sources.size());
        for (int s = 0; s < this.sources.size(); s++) {
            int numLiveNodes = this.liveNodes.get(s).cardinality();
            this.numTotalNodes += numLiveNodes;
            this.numLiveNodesPerSource.add(numLiveNodes);
        }

        maxDegrees = this.sources.stream()
                .max(Comparator.comparingInt(s -> s.maxDegrees().size()))
                .orElseThrow()
                .maxDegrees();
        dimension = this.sources.get(0).getDimension();
        for (var mapper : remappers) {
            maxOrdinal = max(mapper.maxOrdinal(), maxOrdinal);
        }
        this.similarityFunction = similarityFunction;
        this.readerSupplierFactory = readerSupplierFactory;
    }

    /**
     * Validates that all source indexes have compatible configurations and required features
     * before attempting compaction. Ensures consistent dimensions, max degrees, hierarchical
     * settings, and feature sets across all sources.
     */
    private void checkBeforeCompact(
            List<OnDiskGraphIndex> sources,
            List<FixedBitSet> liveNodes,
            List<OrdinalMapper> remappers) {
        validateInputSizes(sources, liveNodes, remappers);
        validateLiveNodesBounds(sources, liveNodes);
        validateGraphConfiguration(sources);
        validateFeatures(sources);
    }

    /**
     * Validates that input lists have consistent sizes and are non-null.
     */
    private void validateInputSizes(List<OnDiskGraphIndex> sources,
                                    List<FixedBitSet> liveNodes,
                                    List<OrdinalMapper> remappers) {
        if (sources.size() < 2) {
            throw new IllegalArgumentException("Must have at least two sources");
        }
        Objects.requireNonNull(liveNodes, "liveNodes");
        Objects.requireNonNull(remappers, "remappers");

        if (sources.size() != liveNodes.size()) {
            throw new IllegalArgumentException("sources and liveNodes must have the same size");
        }
        if (sources.size() != remappers.size()) {
            throw new IllegalArgumentException("sources and remappers must have the same size");
        }
    }

    /**
     * Validates that liveNodes bitsets match the size of their corresponding sources.
     */
    private void validateLiveNodesBounds(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes) {
        for (int s = 0; s < sources.size(); ++s) {
            if (liveNodes.get(s).length() != sources.get(s).getIdUpperBound()) {
                throw new IllegalArgumentException("source " + s + " out of bounds: liveNodes length "
                        + liveNodes.get(s).length() + " != idUpperBound " + sources.get(s).getIdUpperBound());
            }
        }
    }

    /**
     * Validates that all sources have consistent graph configuration (dimensions, degrees, hierarchy).
     */
    private void validateGraphConfiguration(List<OnDiskGraphIndex> sources) {
        int dimension = sources.get(0).getDimension();
        var refDegrees = sources.stream()
                .max(Comparator.comparingInt(s -> s.maxDegrees().size()))
                .orElseThrow()
                .maxDegrees();
        var addHierarchy = sources.get(0).isHierarchical();

        for (OnDiskGraphIndex source : sources) {
            if (source.getDimension() != dimension) {
                throw new IllegalArgumentException("sources must have the same dimension");
            }
            int sharedLevels = Math.min(refDegrees.size(), source.maxDegrees().size());
            for (int d = 0; d < sharedLevels; d++) {
                if (!Objects.equals(source.maxDegrees().get(d), refDegrees.get(d))) {
                    throw new IllegalArgumentException("sources must have the same max degrees");
                }
            }
            if (addHierarchy != source.isHierarchical()) {
                throw new IllegalArgumentException("sources must have the same hierarchical setting");
            }
        }
    }

    /**
     * Validates that all sources have compatible features for compaction.
     */
    private void validateFeatures(List<OnDiskGraphIndex> sources) {
        Set<FeatureId> refKeys = sources.get(0).getFeatures().keySet();
        boolean sameFeatures = sources.stream()
                .skip(1)
                .map(s -> s.getFeatures().keySet())
                .allMatch(refKeys::equals);

        if (!sameFeatures) {
            throw new IllegalArgumentException("Each source must have the same features");
        }
        if (!refKeys.contains(FeatureId.INLINE_VECTORS)) {
            throw new IllegalArgumentException("Each source must have the INLINE_VECTORS feature");
        }
    }

    /**
     * Main compaction entry point. Merges all source indexes into a single output index at the
     * specified path, handling PQ retraining if needed, and writing header, all layers, and footer.
     *
     * <p>Equivalent to {@code compact(outputPath, null)}.
     */
    public void compact(Path outputPath) throws FileNotFoundException {
        compact(outputPath, null);
    }

    /**
     * Main compaction entry point with optional progress reporting. Merges all source indexes into
     * a single output index at the specified path, handling PQ retraining if needed, and writing
     * header, all layers, and footer.
     *
     * @param outputPath       destination file for the merged graph
     * @param progressListener optional callback invoked every ten batches (and at completion of
     *                         each level) so the caller can track I/O progress; pass {@code null}
     *                         to disable progress reporting
     */
    public void compact(Path outputPath, CompactionProgressListener progressListener) throws FileNotFoundException {
        boolean fusedPQEnabled = hasFusedPQ();
        boolean compressedPrecision = fusedPQEnabled;

        ProductQuantization pq;
        int pqLength;
        if (fusedPQEnabled) {
            pq = resolvePQFromSources(similarityFunction);
            pqLength = pq.compressedVectorSize();
        } else {
            pq = null;
            pqLength = -1;
        }

        List<CommonHeader.LayerInfo> layerInfo = computeLayerInfoFromSources();
        int[] entryNodeSource = resolveEntryNodeSource(); // {sourceIdx, originalOrdinal}
        int entryNode = remappers.get(entryNodeSource[0]).oldToNew(entryNodeSource[1]);

        log.info("Writing compacted graph : {} total nodes, maxOrdinal={}, dimension={}, degree={}",
                numTotalNodes, maxOrdinal, dimension, maxDegrees.get(0));
        try (CompactWriter writer = new CompactWriter(outputPath, maxOrdinal, numTotalNodes, 0, layerInfo, entryNode, dimension, maxDegrees, pq, pqLength, fusedPQEnabled)) {
            writer.writeHeader();
            compactLevels(writer, similarityFunction, fusedPQEnabled, compressedPrecision, pq, progressListener);

            // When FusedPQ is enabled and there is no hierarchy (only L0), the reader expects
            // to find the entry node's own PQ code written after the L0 block, just as
            // AbstractGraphIndexWriter.writeSparseLevels does in its getMaxLevel == 0 branch.
            // Without it, loadInMemoryFeatures reads garbage and hierarchyCachedFeatures is
            // missing the entry node, causing "Node X is not in the hierarchy" on first search.
            if (fusedPQEnabled && maxDegrees.size() == 1) {
                try (var entryView = sources.get(entryNodeSource[0]).getView()) {
                    var entryVec = vectorTypeSupport.createFloatVector(dimension);
                    entryView.getVectorInto(entryNodeSource[1], entryVec, 0);
                    var entryPqCode = vectorTypeSupport.createByteSequence(pq.getSubspaceCount());
                    entryPqCode.zero();
                    pq.encodeTo(entryVec, entryPqCode);
                    writer.setEntryNodePqCode(entryPqCode);
                }
            }

            writer.writeFooter();
            log.info("Compaction complete: {}", outputPath);
        } catch (IOException | ExecutionException | InterruptedException e) {
            throw new RuntimeException(e);
        } finally {
            if (ownsExecutor) executor.shutdown();
        }
    }

    /**
     * Returns {sourceIdx, originalOrdinal} for the entry node of the compacted graph.
     * The chosen node must exist at maxLevel (since the on-disk format sets entryNode.level =
     * maxLevel). Prefers the designated entry node of any source whose maxLevel equals the global
     * maxLevel; if all such entry nodes are deleted, falls back to the first live node at maxLevel
     * across all sources.
     */
    private int[] resolveEntryNodeSource() {
        int maxLevel = sources.stream().mapToInt(OnDiskGraphIndex::getMaxLevel).max().orElse(0);

        // The on-disk format sets entryNode.level = layerInfo.size() - 1 (i.e. maxLevel).
        // So the chosen node must actually have neighbors written at maxLevel — meaning it
        // must exist at maxLevel in its source.  Prefer the designated entry node of a
        // maxLevel source; fall back to any live node that is at maxLevel.
        for (int s = 0; s < sources.size(); s++) {
            if (sources.get(s).getMaxLevel() == maxLevel) {
                // try-with-resources: getView() opens a fresh RandomAccessReader that must
                // be closed, otherwise it leaks for the lifetime of the underlying storage.
                int originalEntry;
                try (var entryView = sources.get(s).getView()) {
                    originalEntry = entryView.entryNode().node;
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                if (liveNodes.get(s).get(originalEntry)) {
                    return new int[]{s, originalEntry};
                }
            }
        }

        // Entry nodes were all deleted: scan for any live node that exists at maxLevel.
        for (int s = 0; s < sources.size(); s++) {
            if (sources.get(s).getMaxLevel() < maxLevel) continue;
            NodesIterator it = sources.get(s).getNodes(maxLevel);
            while (it.hasNext()) {
                int node = it.next();
                if (liveNodes.get(s).get(node)) {
                    return new int[]{s, node};
                }
            }
        }

        throw new IllegalStateException("No live nodes found at maxLevel=" + maxLevel);
    }

    /**
     * Compacts all hierarchical levels of the graph, processing each level in batches.
     * For level 0 (base layer), writes inline vectors and neighbors. For upper layers,
     * writes only graph structure and optional PQ codes.
     */
    private void compactLevels(CompactWriter writer,
                                 VectorSimilarityFunction similarityFunction,
                                 boolean fusedPQEnabled,
                                 boolean compressedPrecision,
                                 ProductQuantization pq,
                                 CompactionProgressListener progressListener)
            throws IOException, ExecutionException, InterruptedException {

        int maxUpperDegree = 0;
        for (int level = 1; level < maxDegrees.size(); level++) {
            maxUpperDegree = Math.max(maxUpperDegree, maxDegrees.get(level));
        }

        int baseSearchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(0) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0);
        int upperMaxPerSourceTopK = maxUpperDegree == 0 ? 0 : Math.max(MIN_SEARCH_TOP_K, ((maxUpperDegree + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int upperMaxCandidateSize = upperMaxPerSourceTopK * sources.size();
        int maxCandidateSize = Math.max(baseMaxCandidateSize, upperMaxCandidateSize);
        int scratchDegree = Math.max(maxDegrees.get(0), Math.max(1, maxUpperDegree));
        // Batch tasks run on `executor`, whose worker threads each lazily create their own
        // Scratch via this ThreadLocal. compactLevels() cannot reach those worker-thread
        // ThreadLocal entries with threadLocalScratch.get()/remove(), so every Scratch is
        // registered in `allScratch` and closed in the finally block below. Closing only the
        // calling thread's Scratch leaks every worker thread's GraphSearchers — and the
        // RandomAccessReaders behind their Views — for the lifetime of the executor.
        final Queue<Scratch> allScratch = new ConcurrentLinkedQueue<>();
        final ThreadLocal<Scratch> threadLocalScratch = ThreadLocal.withInitial(() -> {
            Scratch scratch = new Scratch(maxCandidateSize, scratchDegree, dimension, sources, pq);
            allScratch.add(scratch);
            return scratch;
        });
        // Every batch task submitted to `executor` is tracked here so the finally block can
        // quiesce the executor (cancel + await) before draining `allScratch`. Without this
        // barrier, when compaction fails an in-flight task that has not yet run the
        // ThreadLocal supplier could register a fresh Scratch AFTER the drain — leaking it.
        final Queue<Future<?>> submittedFutures = new ConcurrentLinkedQueue<>();

        boolean compactionSucceeded = false;
        try {
            for (int level = 0; level < maxDegrees.size(); level++) {
                List<BatchSpec> batches = buildBatches(level);
                int searchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(level) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
                int beamWidth = Math.max(maxDegrees.get(level), searchTopK) * BEAM_WIDTH_MULTIPLIER;

                CompactionParams params = new CompactionParams(fusedPQEnabled, compressedPrecision, searchTopK, beamWidth, pq);

                if (level == 0) {
                    log.info("Compacting level 0 (base layer)");

                    ExecutorCompletionService<List<WriteResult>> ecs =
                            new ExecutorCompletionService<>(executor);

                    java.util.function.Consumer<BatchSpec> submitOne = (bs) -> {
                        submittedFutures.add(ecs.submit(() -> {
                            Scratch scratch = threadLocalScratch.get();
                            return computeBaseBatch(writer, bs, scratch, params);
                        }));
                    };

                    var wropts = EnumSet.of(StandardOpenOption.WRITE, StandardOpenOption.READ);
                    try (FileChannel fc = FileChannel.open(writer.getOutputPath(), wropts)) {
                        writer.setInlineChannel(fc);
                        try {
                            runBatchesWithBackpressure(
                                    batches,
                                    ecs,
                                    submitOne,
                                    (results) -> { /* records were written directly by workers */ },
                                    progressListener
                            );
                        } finally {
                            writer.setInlineChannel(null);
                        }
                    }

                    writer.offsetAfterInline();

                } else {
                    final int lvl = level;
                    log.info("Compacting upper layer {}", level);

                    ExecutorCompletionService<List<UpperLayerWriteResult>> ecs =
                            new ExecutorCompletionService<>(executor);

                    java.util.function.Consumer<BatchSpec> submitOne = (bs) -> {
                        submittedFutures.add(ecs.submit(() -> {
                            Scratch scratch = threadLocalScratch.get();
                            return computeUpperBatchForLevel(bs, lvl, scratch, params);
                        }));
                    };

                    runBatchesWithBackpressure(
                            batches,
                            ecs,
                            submitOne,
                            (results) -> {
                                try {
                                    for (UpperLayerWriteResult r : results) {
                                        writer.writeUpperLayerNode(
                                                lvl,
                                                r.ordinal,
                                                r.neighbors,
                                                r.pqCode
                                        );
                                    }
                                } catch (IOException e) {
                                    throw new UncheckedIOException(e);
                                }
                            },
                            progressListener
                    );
                }
            }
            compactionSucceeded = true;
        } finally {
            // Quiesce the executor BEFORE draining allScratch: until every submitted task
            // has terminated, an in-flight task can still run the threadLocalScratch
            // supplier and register a fresh Scratch that the drain below would miss.
            awaitAllTasks(submittedFutures);
            // Drop this thread's ThreadLocal entry; worker-thread entries are unreachable
            // here but their Scratch instances are all registered in `allScratch`.
            threadLocalScratch.remove();
            if (compactionSucceeded) {
                // No primary exception in flight, so a close failure can be surfaced safely.
                closeAllScratch(allScratch);
            } else {
                // A primary exception is already propagating; close best-effort so a
                // secondary close failure does not mask the original cause.
                closeAllScratchQuietly(allScratch);
            }
        }
    }

    /**
     * Cancels and waits for every batch task submitted to {@code executor}. This MUST run
     * before {@code allScratch} is drained: until a submitted task has terminated it can
     * still execute the {@code threadLocalScratch} supplier and {@code allScratch.add(...)}
     * a fresh Scratch, which the drain would then miss and leak.
     *
     * <p>On the normal-completion path every task is already done, so {@code cancel} is a
     * no-op and {@code get} returns immediately — a cheap barrier. On the failure path the
     * not-yet-started tasks are cancelled (they never create a Scratch) and the running
     * ones are awaited (their Scratch is registered and will be drained).
     */
    private static void awaitAllTasks(Queue<Future<?>> submittedFutures) {
        boolean interrupted = false;
        Future<?> f;
        while ((f = submittedFutures.poll()) != null) {
            // No-op for already-completed tasks; interrupts the rest so a failed compaction
            // need not wait for the whole sliding window to drain naturally.
            f.cancel(true);
            if (interrupted) {
                // Already interrupted: keep cancelling the remainder but do not block.
                continue;
            }
            try {
                f.get();
            } catch (CancellationException | ExecutionException ignored) {
                // We only need the task to have TERMINATED; its result/failure was already
                // handled by runBatchesWithBackpressure (or is irrelevant once cancelled).
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Closes every {@link Scratch} created during compaction, regardless of which thread
     * created it. A close failure on one Scratch does not prevent the others from being
     * closed; the first failure is rethrown with any later ones attached as suppressed.
     */
    private static void closeAllScratch(Queue<Scratch> allScratch) throws IOException {
        Throwable firstError = null;
        Scratch scratch;
        while ((scratch = allScratch.poll()) != null) {
            try {
                scratch.close();
            } catch (IOException | RuntimeException e) {
                // Catch RuntimeException too: a single Scratch failing to close must not
                // abort the loop, otherwise every remaining Scratch would leak.
                if (firstError == null) {
                    firstError = e;
                } else {
                    firstError.addSuppressed(e);
                }
            }
        }
        if (firstError instanceof IOException) {
            throw (IOException) firstError;
        } else if (firstError instanceof RuntimeException) {
            throw (RuntimeException) firstError;
        }
    }

    /**
     * Like {@link #closeAllScratch} but never throws — used when a primary exception is
     * already propagating, so a close failure is logged instead of masking the cause.
     */
    private static void closeAllScratchQuietly(Queue<Scratch> allScratch) {
        Scratch scratch;
        while ((scratch = allScratch.poll()) != null) {
            try {
                scratch.close();
            } catch (IOException | RuntimeException e) {
                // Broad-ish catch on purpose: one bad close must not abort the rest.
                log.warn("Failed to close compaction Scratch space", e);
            }
        }
    }

    /**
     * Divides nodes at a given level across all source indexes into processing batches
     * for parallel execution. Each batch contains a subset of nodes from one source.
     */
    private List<BatchSpec> buildBatches(int level) {
        List<BatchSpec> batches = new ArrayList<>();

        for (int s = 0; s < sources.size(); ++s) {
            var source = sources.get(s);
            if (level > source.getMaxLevel()) continue;
            NodesIterator sourceNodes = source.getNodes(level);
            int numNodes = sourceNodes.size();
            int[] nodes = new int[numNodes];
            int i = 0;
            while (sourceNodes.hasNext()) {
                nodes[i++] = sourceNodes.next();
            }

            int numBatches = max(TARGET_BATCHES_PER_SOURCE, (numNodes + TARGET_NODES_PER_BATCH - 1) / TARGET_NODES_PER_BATCH);
            if (numBatches > numNodes) numBatches = numNodes;
            int batchSize = (numNodes + numBatches - 1) / numBatches;
            for (int b = 0; b < numBatches; ++b) {
                int start = min(numNodes, batchSize * b);
                int end = min(numNodes, batchSize * (b + 1));
                batches.add(new BatchSpec(s, nodes, start, end));
            }
        }

        return batches;
    }

    /**
     * Processes a batch of base layer (level 0) nodes from one source index. For each live node,
     * gathers candidates from all sources, applies diversity selection, and creates write results
     * containing the full node record data.
     */
   private List<WriteResult> computeBaseBatch(CompactWriter writer,
                                              BatchSpec bs,
                                              Scratch scratch,
                                              CompactionParams params) throws IOException {

        List<WriteResult> out = new ArrayList<>(bs.end - bs.start);

        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];
            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;

            out.add(processBaseNode(node, bs.sourceIdx, scratch, writer, params));
        }

        return out;
    }

    /**
     * Processes a batch of upper layer nodes from one source index. Similar to base layer
     * processing but returns only ordinal, neighbors, and optional PQ code (no inline vectors).
     */
    private List<UpperLayerWriteResult> computeUpperBatchForLevel(
            BatchSpec bs,
            int level,
            Scratch scratch,
            CompactionParams params
    ) {
        List<UpperLayerWriteResult> results =
                new ArrayList<>(bs.end - bs.start);

        for (int i = bs.start; i < bs.end; i++) {
            int node = bs.nodes[i];

            if (!liveNodes.get(bs.sourceIdx).get(node)) continue;

            results.add(processUpperNode(node, bs.sourceIdx, level, scratch, params));
        }

        return results;
    }

    /**
     * Processes a single base layer node: retrieves its vector, gathers diverse candidates from
     * all sources, selects best neighbors using diversity criteria, remaps ordinals, and returns
     * the complete write result for this node.
     */
    private WriteResult processBaseNode(
            int node,
            int sourceIdx,
            Scratch scratch,
            CompactWriter writer,
            CompactionParams params
    ) throws IOException {

        var sourceView = (OnDiskGraphIndex.View) scratch.gs[sourceIdx].getView();
        sourceView.getVectorInto(node, scratch.baseVec, 0);

        int candSize = gatherCandidates(node, 0, sourceIdx, scratch, scratch.baseVec, params);

        int[] order = IntStream.range(0, candSize).toArray();
        sortOrderByScoreDesc(order, scratch.candScore, candSize);

        var selected = scratch.selectedCache;

        new CompactVamanaDiversityProvider(similarityFunction, 1.2f)
                .retainDiverse(
                        scratch.candSrc,
                        scratch.candNode,
                        scratch.candScore,
                        order,
                        candSize,
                        maxDegrees.get(0),
                        selected,
                        scratch.tmpVec,
                        scratch.gs
                );

        // remap
        for (int k = 0; k < selected.size; k++) {
            selected.nodes[k] =
                    remappers.get(selected.sourceIdx[k])
                            .oldToNew(selected.nodes[k]);
        }

        int newOrdinal = remappers.get(sourceIdx).oldToNew(node);

        return writer.writeInlineNodeRecord(
                newOrdinal,
                scratch.baseVec,
                scratch.tmpVec,
                selected,
                scratch.pqCode
        );
    }

    /**
     * Processes a single upper layer node: similar to base layer processing but only returns
     * graph structure (ordinal and neighbors) and optional PQ encoding for level 1.
     */
    private UpperLayerWriteResult processUpperNode(
            int node,
            int sourceIdx,
            int level,
            Scratch scratch,
            CompactionParams params
    ) {
        var sourceView = (OnDiskGraphIndex.View) scratch.gs[sourceIdx].getView();
        sourceView.getVectorInto(node, scratch.baseVec, 0);

        int candSize = gatherCandidates(node, level, sourceIdx, scratch, scratch.baseVec, params);

        int[] order = IntStream.range(0, candSize).toArray();
        sortOrderByScoreDesc(order, scratch.candScore, candSize);

        var selected = scratch.selectedCache;

        new CompactVamanaDiversityProvider(similarityFunction, 1.2f)
                .retainDiverse(
                        scratch.candSrc,
                        scratch.candNode,
                        scratch.candScore,
                        order,
                        candSize,
                        maxDegrees.get(level),
                        selected,
                        scratch.tmpVec,
                        scratch.gs
                );

        // remap
        for (int k = 0; k < selected.size; k++) {
            selected.nodes[k] =
                    remappers.get(selected.sourceIdx[k])
                            .oldToNew(selected.nodes[k]);
        }

        int newOrdinal = remappers.get(sourceIdx).oldToNew(node);

        ByteSequence<?> pqCode = maybeEncodePQ(level, scratch, params);

        return new UpperLayerWriteResult(newOrdinal, selected, pqCode);
    }

    /**
     * Encodes a vector using Product Quantization if enabled and the level is 1.
     * Returns null otherwise.
     */
    private ByteSequence<?> maybeEncodePQ(int level, Scratch scratch, CompactionParams params) {
        if (!params.fusedPQEnabled || level != 1) {
            return null;
        }

        scratch.pqCode.zero();
        params.pq.encodeTo(scratch.baseVec, scratch.pqCode);
        return scratch.pqCode.copy();
    }

    /**
     * Collects neighbor candidates for a node from all source indexes. For the source containing
     * the node, uses existing neighbors; for other sources, performs graph search. Returns the
     * total number of candidates gathered.
     */
    private int gatherCandidates(
            int node,
            int level,
            int sourceIdx,
            Scratch scratch,
            VectorFloat<?> baseVec,
            CompactionParams params
    ) {
        int candSize = 0;

        for (int ss = 0; ss < sources.size(); ss++) {
            var searchView = (OnDiskGraphIndex.View) scratch.gs[ss].getView();
            var indexAlive = liveNodes.get(ss);

            if (ss == sourceIdx) {
                candSize = gatherFromSameSource(node, level, ss, searchView, indexAlive,
                                                 baseVec, scratch, candSize);
            } else {
                candSize = gatherFromOtherSource(node, level, ss, searchView, indexAlive,
                                                  baseVec, scratch, candSize, params);
            }
        }

        return candSize;
    }

    /**
     * Gathers candidates from the same source index that contains the node.
     * Simply iterates through existing neighbors.
     */
    private int gatherFromSameSource(int node, int level, int sourceIdx,
                                     OnDiskGraphIndex.View searchView, FixedBitSet indexAlive,
                                     VectorFloat<?> baseVec, Scratch scratch, int candSize) {
        var it = searchView.getNeighborsIterator(level, node);
        while (it.hasNext()) {
            int nb = it.nextInt();
            if (!indexAlive.get(nb)) continue;

            searchView.getVectorInto(nb, scratch.tmpVec, 0);

            scratch.candSrc[candSize] = sourceIdx;
            scratch.candNode[candSize] = nb;
            scratch.candScore[candSize] = similarityFunction.compare(baseVec, scratch.tmpVec);
            candSize++;
        }
        return candSize;
    }

    /**
     * Gathers candidates from a different source index via graph search.
     */
    private int gatherFromOtherSource(int node, int level, int sourceIdx,
                                      OnDiskGraphIndex.View searchView, FixedBitSet indexAlive,
                                      VectorFloat<?> baseVec, Scratch scratch, int candSize,
                                      CompactionParams params) {
        SearchScoreProvider ssp = buildCrossSourceScoreProvider(
                params.compressedPrecision,
                sources.get(sourceIdx),
                searchView,
                baseVec,
                scratch.tmpVec,
                similarityFunction
        );

        if (level == 0) {
            SearchResult results = scratch.gs[sourceIdx].search(
                    ssp, params.searchTopK, params.beamWidth, 0f, 0f, indexAlive
            );

            for (var r : results.getNodes()) {
                scratch.candSrc[candSize] = sourceIdx;
                scratch.candNode[candSize] = r.node;
                scratch.candScore[candSize] =
                        params.fusedPQEnabled
                                ? rescore(searchView, r.node, baseVec, scratch.tmpVec)
                                : r.score;
                candSize++;
            }
        } else {
            var entry = searchView.entryNode();
            if (level > entry.level) return candSize;
            scratch.gs[sourceIdx].initializeInternal(ssp, entry, Bits.ALL);

            // Descend greedily through levels above the target level, so the search at
            // `level` starts from the best-known region rather than the global entry node.
            // This mirrors how GraphSearcher.searchInternal navigates the hierarchy.
            for (int l = entry.level; l > level; l--) {
                scratch.gs[sourceIdx].searchOneLayer(ssp, 1, 0f, l, Bits.ALL);
                scratch.gs[sourceIdx].setEntryPointsFromPreviousLayer();
            }

            scratch.gs[sourceIdx].searchOneLayer(
                    ssp, params.searchTopK, 0f, level, indexAlive
            );

            int prev_candSize = candSize;
            candSize = appendApproximateResults(
                    scratch.gs[sourceIdx].approximateResults,
                    sourceIdx,
                    scratch,
                    candSize
            );

            if (params.fusedPQEnabled) {
                for (int i = prev_candSize; i < candSize; i++) {
                    scratch.candScore[i] = rescore(
                            searchView,
                            scratch.candNode[i],
                            baseVec,
                            scratch.tmpVec
                    );
                }
            }
        }

        return candSize;
    }

    /**
     * Recomputes exact similarity score between the base vector and a node's vector,
     * used to refine approximate PQ-based search results.
     */
    private float rescore(OnDiskGraphIndex.View view,
                         int node,
                         VectorFloat<?> base,
                         VectorFloat<?> tmp) {
        view.getVectorInto(node, tmp, 0);
        return similarityFunction.compare(base, tmp);
    }

    /**
     * Executes batches with controlled concurrency using a sliding window approach. Prevents
     * overwhelming memory by limiting the number of in-flight tasks while maintaining high
     * throughput via the completion service.
     *
     * @param progressListener optional; when non-null, called every ten batches (and at the
     *                         final batch) with {@code (completedBatches, totalBatches)} so
     *                         the caller can expose live compaction progress
     */
    private <T> void runBatchesWithBackpressure(
            List<BatchSpec> batches,
            ExecutorCompletionService<List<T>> ecs,
            java.util.function.Consumer<BatchSpec> submitOne,
            java.util.function.Consumer<List<T>> onComplete,
            CompactionProgressListener progressListener
    ) throws InterruptedException, ExecutionException {

        final int total = batches.size();
        int nextToSubmit = 0;
        int inFlight = 0;

        // initial window
        while (inFlight < taskWindowSize && nextToSubmit < total) {
            submitOne.accept(batches.get(nextToSubmit++));
            inFlight++;
        }

        int completed = 0;
        while (completed < total) {
            List<T> results = ecs.take().get();
            onComplete.accept(results);

            completed++;
            inFlight--;

            if (nextToSubmit < total) {
                submitOne.accept(batches.get(nextToSubmit++));
                inFlight++;
            }
            if (completed % 10 == 0 || completed == total) {
                log.info("Compaction I/O progress: {}/{} batches written to disk", completed, total);
                if (progressListener != null) {
                    progressListener.onProgress(completed, total);
                }
            }
        }
    }

    /**
     * Appends search results from a NodeQueue to the candidate arrays, returning the updated
     * candidate count.
     */
    private int appendApproximateResults(NodeQueue queue,
                                         int sourceIdx,
                                         Scratch scratch,
                                         int candSize) {
        final int ss = sourceIdx;
        final int[] idx = new int[] { candSize };

        queue.foreach((nb, score) -> {
            scratch.candSrc[idx[0]] = ss;
            scratch.candNode[idx[0]] = nb;
            scratch.candScore[idx[0]] = score;
            idx[0]++;
        });

        return idx[0];
    }

    /**
     * Computes layer metadata for the compacted graph by counting live nodes at each level
     * across all source indexes.
     */
    private List<CommonHeader.LayerInfo> computeLayerInfoFromSources() {
        int maxLevel = sources.stream().mapToInt(OnDiskGraphIndex::getMaxLevel).max().orElse(0);
        List<CommonHeader.LayerInfo> layerInfo = new ArrayList<>(maxLevel + 1);
        for (int level = 0; level <= maxLevel; level++) {
            int count = 0;
            for (int s = 0; s < sources.size(); s++) {
                if (level > sources.get(s).getMaxLevel()) continue;
                NodesIterator it = sources.get(s).getNodes(level);
                FixedBitSet alive = liveNodes.get(s);
                while (it.hasNext()) {
                    int node = it.next();
                    if (alive.get(node)) count++;
                }
            }
            layerInfo.add(new CommonHeader.LayerInfo(count, maxDegrees.get(level)));
        }
        return layerInfo;
    }

    /**
     * Trains a new Product Quantization codebook using balanced sampling across all source
     * indexes. This ensures the PQ is optimized for the combined dataset.
     */
    private ProductQuantization resolvePQFromSources(VectorSimilarityFunction similarityFunction) {
        // Pass the reader-supplier factory so PQRetrainer can open bulk sequential readers
        // instead of per-node block-cache reads when remote storage is involved (issue #599).
        PQRetrainer retrainer = new PQRetrainer(sources, liveNodes, dimension, readerSupplierFactory);
        return retrainer.retrain(similarityFunction);
    }

    /**
     * Checks if the source indexes have FusedPQ feature enabled.
     */
    private boolean hasFusedPQ() {
        return sources.get(0).getFeatures().containsKey(FeatureId.FUSED_PQ);
    }

    /**
     * Creates a score provider for searching across different source indexes. Uses approximate
     * PQ-based scoring if compressedPrecision is enabled, otherwise uses exact scoring.
     */
    private SearchScoreProvider buildCrossSourceScoreProvider(boolean compressedPrecision,
                                                              OnDiskGraphIndex searchSource,
                                                              OnDiskGraphIndex.View searchView,
                                                              VectorFloat<?> baseVec,
                                                              VectorFloat<?> tmpVec,
                                                              VectorSimilarityFunction similarityFunction) {
        if (compressedPrecision) {
            ScoreFunction.ExactScoreFunction reranker =
                node2 -> {
                    searchView.getVectorInto(node2, tmpVec, 0);
                    return similarityFunction.compare(baseVec, tmpVec);
                };
            var asf = ((FusedPQ) searchSource.getFeatures().get(FeatureId.FUSED_PQ)).approximateScoreFunctionFor(baseVec, similarityFunction, searchView, reranker);

            return new DefaultSearchScoreProvider(asf);
        }

        var sf = new ScoreFunction.ExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                searchView.getVectorInto(node2, tmpVec, 0);
                return similarityFunction.compare(baseVec, tmpVec);
            }
        };
        return new DefaultSearchScoreProvider(sf);
    }

    /**
     * Estimates the RAM usage of this compactor instance.
     * Accounts for data structures used during compaction including bitsets, remappers,
     * executor overhead, and per-thread scratch space.
     */
    @Override
    public long ramBytesUsed() {
        int OH = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF = RamUsageEstimator.NUM_BYTES_OBJECT_REF;

        // Shallow size of this object (header + fields)
        // Current fields: sources, liveNodes, numLiveNodesPerSource, remappers, maxDegrees,
        //                dimension(int), maxOrdinal(int), numTotalNodes(int),
        //                ownsExecutor(boolean), executor, taskWindowSize(int), similarityFunction
        long size = OH + 8L * REF + Integer.BYTES * 4 + 1;

        // liveNodes: FixedBitSet per source
        for (var entry : liveNodes) {
            size += entry.ramBytesUsed();
        }

        // numLiveNodesPerSource: ArrayList of Integers
        size += OH + REF + (long) numLiveNodesPerSource.size() * (OH + Integer.BYTES);

        // remappers: each MapMapper holds an oldToNew HashMap and newToOld Int2IntHashMap
        // Estimate based on the number of mappings
        for (var mapper : remappers) {
            // Object overhead + two maps with int key/value pairs
            // HashMap entry: ~32 bytes each; Int2IntHashMap: ~16 bytes per entry
            if (mapper instanceof OrdinalMapper.MapMapper) {
                // rough estimate: the mapper stores two maps over all mapped ordinals
                size += OH + (long) (maxOrdinal + 1) * 48;
            }
        }

        // maxDegrees: small list of integers
        size += OH + REF + (long) maxDegrees.size() * (OH + Integer.BYTES);

        // executor: ForkJoinPool overhead (if owned)
        // Estimate based on number of threads
        int numThreads = ownsExecutor ? Runtime.getRuntime().availableProcessors() : taskWindowSize;
        if (ownsExecutor) {
            size += OH + REF;
        }

        // Scratch space: ThreadLocal instances (one per active thread)
        // Each Scratch contains:
        //   - candSrc, candNode, candScore arrays
        //   - SelectedVecCache (with its own arrays and vector copies)
        //   - tmpVec, baseVec (VectorFloat instances)
        //   - GraphSearcher array (one per source)
        //   - pqCode ByteSequence
        size += estimateScratchSpacePerThread() * numThreads;

        return size;
    }

    /**
     * Estimates the RAM usage of a single Scratch instance.
     */
    private long estimateScratchSpacePerThread() {
        int OH = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF = RamUsageEstimator.NUM_BYTES_OBJECT_REF;

        // Calculate maxCandidateSize and maxDegree (same logic as in compactLevels)
        int maxUpperDegree = 0;
        for (int level = 1; level < maxDegrees.size(); level++) {
            maxUpperDegree = Math.max(maxUpperDegree, maxDegrees.get(level));
        }
        int baseSearchTopK = Math.max(MIN_SEARCH_TOP_K, ((maxDegrees.get(0) + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int baseMaxCandidateSize = baseSearchTopK * (sources.size() - 1) + maxDegrees.get(0);
        int upperMaxPerSourceTopK = maxUpperDegree == 0 ? 0 : Math.max(MIN_SEARCH_TOP_K, ((maxUpperDegree + sources.size() - 1) / sources.size()) * SEARCH_TOP_K_MULTIPLIER);
        int upperMaxCandidateSize = upperMaxPerSourceTopK * sources.size();
        int maxCandidateSize = Math.max(baseMaxCandidateSize, upperMaxCandidateSize);
        int scratchDegree = Math.max(maxDegrees.get(0), Math.max(1, maxUpperDegree));

        long scratchSize = OH + 6L * REF;

        // candSrc, candNode, candScore arrays
        scratchSize += (long) maxCandidateSize * Integer.BYTES; // candSrc
        scratchSize += (long) maxCandidateSize * Integer.BYTES; // candNode
        scratchSize += (long) maxCandidateSize * Float.BYTES;   // candScore

        // SelectedVecCache
        scratchSize += OH + 5L * REF + Integer.BYTES; // SelectedVecCache object
        scratchSize += (long) scratchDegree * Integer.BYTES;  // sourceIdx array
        scratchSize += (long) scratchDegree * REF;            // views array
        scratchSize += (long) scratchDegree * Integer.BYTES;  // nodes array
        scratchSize += (long) scratchDegree * Float.BYTES;    // scores array
        scratchSize += (long) scratchDegree * REF;            // vecs array
        scratchSize += (long) scratchDegree * (OH + dimension * Float.BYTES); // VectorFloat instances

        // tmpVec and baseVec
        scratchSize += 2L * (OH + dimension * Float.BYTES);

        // GraphSearcher array (one per source)
        scratchSize += (long) sources.size() * REF;
        // Each GraphSearcher has internal state - rough estimate
        scratchSize += (long) sources.size() * (OH + 10L * REF);

        // pqCode ByteSequence (if PQ enabled)
        if (hasFusedPQ()) {
            FusedPQ fpq = (FusedPQ) sources.get(0).getFeatures().get(FeatureId.FUSED_PQ);
            int subspaceCount = fpq.getPQ().getSubspaceCount();
            scratchSize += OH + subspaceCount; // ByteSequence
        }

        return scratchSize;
    }

    /**
     * Encapsulates common parameters used throughout the compaction process.
     */
    private static final class CompactionParams {
        final boolean fusedPQEnabled;
        final boolean compressedPrecision;
        final int searchTopK;
        final int beamWidth;
        final ProductQuantization pq;

        CompactionParams(boolean fusedPQEnabled, boolean compressedPrecision,
                        int searchTopK, int beamWidth, ProductQuantization pq) {
            this.fusedPQEnabled = fusedPQEnabled;
            this.compressedPrecision = compressedPrecision;
            this.searchTopK = searchTopK;
            this.beamWidth = beamWidth;
            this.pq = pq;
        }
    }

    /**
     * Sorts an index array by descending score values using quicksort.
     */
    private static void sortOrderByScoreDesc(int[] order, float[] score, int size) {
        quicksort(order, score, 0, size - 1);
    }

    /**
     * Tail-recursive quicksort implementation for sorting by score in descending order.
     */
    private static void quicksort(int[] order, float[] score, int lo, int hi) {
        while (lo < hi) {
            int p = partition(order, score, lo, hi);
            // recurse smaller side first (limits stack)
            if (p - lo < hi - p) {
                quicksort(order, score, lo, p - 1);
                lo = p + 1;
            } else {
                quicksort(order, score, p + 1, hi);
                hi = p - 1;
            }
        }
    }

    /**
     * Partitions the order array for quicksort using descending score comparison.
     */
    private static int partition(int[] order, float[] score, int lo, int hi) {
        float pivot = score[order[hi]];
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (score[order[j]] > pivot) { // DESC
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
                i++;
            }
        }
        int t = order[i];
        order[i] = order[hi];
        order[hi] = t;
        return i;
    }

    static final class WriteResult {
        final int newOrdinal;
        final long fileOffset;

        WriteResult(int newOrdinal, long fileOffset) {
            this.newOrdinal = newOrdinal;
            this.fileOffset = fileOffset;
        }
    };

    private static final class UpperLayerWriteResult {
        final int ordinal;
        final int[] neighbors;
        final ByteSequence<?> pqCode;

        UpperLayerWriteResult(int ordinal, SelectedVecCache cache, ByteSequence<?> pqCode) {
            this.ordinal = ordinal;
            this.neighbors = Arrays.copyOf(cache.nodes, cache.size);
            this.pqCode = pqCode == null ? null : pqCode.copy();
        }
    };


    /**
     * Thread-local scratch space containing reusable buffers and search state for processing nodes.
     */
    private static final class Scratch implements AutoCloseable {

        final int[] candSrc, candNode;
        final float[] candScore;
        final SelectedVecCache selectedCache;
        final VectorFloat<?> tmpVec, baseVec;
        final GraphSearcher[] gs;
        final ByteSequence<?> pqCode;

        /**
         * Constructs scratch space with buffers sized for the maximum expected candidates and degree.
         */
        Scratch(int maxCandidateSize, int maxDegree, int dimension, List<OnDiskGraphIndex> sources, ProductQuantization pq) {
            this.candSrc = new int[maxCandidateSize];
            this.candNode = new int[maxCandidateSize];
            this.candScore = new float[maxCandidateSize];
            this.selectedCache = new SelectedVecCache(maxDegree, dimension);
            this.tmpVec = vectorTypeSupport.createFloatVector(dimension);
            this.baseVec = vectorTypeSupport.createFloatVector(dimension);
            this.pqCode = (pq == null) ? null : vectorTypeSupport.createByteSequence(pq.getSubspaceCount());

            this.gs = new GraphSearcher[sources.size()];
            // Each GraphSearcher opens a View (hence a RandomAccessReader) on its source.
            // If a later source's getView() throws, close the searchers already created so
            // their readers are not leaked — this Scratch is never returned to the caller,
            // so it would otherwise never be registered for the compactor's drain.
            boolean initialised = false;
            try {
                for (int i = 0; i < sources.size(); i++) {
                    gs[i] = new GraphSearcher(sources.get(i));
                    gs[i].usePruning(false);
                }
                initialised = true;
            } finally {
                if (!initialised) {
                    for (GraphSearcher s : gs) {
                        if (s != null) {
                            try {
                                s.close();
                            } catch (IOException | RuntimeException ignored) {
                                // best-effort cleanup; the original failure propagates
                            }
                        }
                    }
                }
            }
        }

        /**
         * Closes all graph searchers and resets the cache. Every searcher is closed even
         * if one throws, otherwise a single failing GraphSearcher.close() would leak the
         * RandomAccessReaders behind the remaining searchers' Views; the first failure is
         * rethrown with any later ones attached as suppressed.
         */
        @Override
        public void close() throws IOException {
            Throwable firstError = null;
            for (GraphSearcher s : gs) {
                // The constructor may leave a trailing null slot if it failed partway.
                if (s == null) {
                    continue;
                }
                try {
                    s.close();
                } catch (IOException | RuntimeException e) {
                    if (firstError == null) {
                        firstError = e;
                    } else {
                        firstError.addSuppressed(e);
                    }
                }
            }
            selectedCache.reset();
            if (firstError instanceof IOException) {
                throw (IOException) firstError;
            } else if (firstError instanceof RuntimeException) {
                throw (RuntimeException) firstError;
            }
        }
    }

    /**
     * Specification for a batch of nodes to be processed from one source index.
     */
    private static final class BatchSpec {
        final int sourceIdx;
        final int[] nodes;              // materialized node ids for this source
        final int start;
        final int end;

        BatchSpec(int sourceIdx, int[] nodes, int start, int end) {
            this.sourceIdx = sourceIdx;
            this.nodes = nodes;
            this.start = start;
            this.end = end;
        }
    }

    /**
     * Provides Vamana-style diversity filtering for neighbor selection during compaction.
     */
    private static final class CompactVamanaDiversityProvider {
        /**
         * the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more
         */
        public final float alpha;

        /**
         * used to compute diversity
         */
        public final VectorSimilarityFunction vsf;

        /**
         * Create a new diversity provider
         */
        public CompactVamanaDiversityProvider(VectorSimilarityFunction vsf, float alpha) {
            this.vsf = vsf;
            this.alpha = alpha;
        }

        /**
         * Selects diverse neighbors from candidates using gradually increasing alpha threshold.
         * Update `selected` with the diverse members of `neighbors`.  `neighbors` is not modified
         * It assumes that the i-th neighbor with 0 {@literal <=} i {@literal <} diverseBefore is already diverse.
         */
        public void retainDiverse(int[] candSrc, int[] candNode, float[] candScore, int[] order, int orderSize, int maxDegree, SelectedVecCache selectedCache, VectorFloat<?> tmp, GraphSearcher[] gs) {
            selectedCache.reset();
            if (orderSize == 0) return;
            int nSelected = 0;

            // add diverse candidates, gradually increasing alpha to the threshold
            // (so that the nearest candidates are prioritized)
            float currentAlpha = 1.0f;
            while (currentAlpha <= alpha + 1E-6 && nSelected < maxDegree) {
                for (int i = 0; i < orderSize && nSelected < maxDegree; i++) {
                    int ci = order[i];
                    int cSrc = candSrc[ci];
                    int cNode = candNode[ci];
                    float cScore = candScore[ci];

                    OnDiskGraphIndex.View cView = (OnDiskGraphIndex.View) gs[cSrc].getView();
                    cView.getVectorInto(cNode, tmp, 0);
                    if (isDiverse(cView, cNode, tmp, cScore, currentAlpha, selectedCache)) {
                        selectedCache.add(cSrc, cView, cNode, cScore, tmp);
                        nSelected++;
                    }
                }

                currentAlpha += DIVERSITY_ALPHA_STEP;
            }
        }

        /**
         * Checks if a candidate is diverse enough by ensuring it's closer to the base node
         * than to any already-selected neighbor (scaled by alpha threshold).
         */
        private boolean isDiverse(OnDiskGraphIndex.View cView, int cNode, VectorFloat<?> cVec, float cScore, float alpha, SelectedVecCache selectedCache) {
            for (int j = 0; j < selectedCache.size; j++) {
                if (selectedCache.views[j] == cView && selectedCache.nodes[j] == cNode) {
                    return false; // already selected; don't add a duplicate
                }
                if (vsf.compare(cVec, selectedCache.vecs[j]) > cScore * alpha) {
                    return false;
                }
            }
            return true;
        }

    }

    /**
     * Cache for storing selected diverse neighbors along with their metadata and vector copies.
     */
    static final class SelectedVecCache {
        int[] sourceIdx;
        OnDiskGraphIndex.View[] views;
        int[] nodes;
        float[] scores;
        VectorFloat<?>[] vecs;
        int size;

        /**
         * Constructs a cache with the specified capacity and vector dimension.
         */
        SelectedVecCache(int capacity, int dimension) {
            sourceIdx = new int[capacity];
            views = new OnDiskGraphIndex.View[capacity];
            nodes = new int[capacity];
            scores = new float[capacity];
            vecs = new VectorFloat<?>[capacity];
            for(int c = 0; c < capacity; ++c) {
                vecs[c] = vectorTypeSupport.createFloatVector(dimension);
            }
            size = 0;
        }

        /**
         * Resets the cache for reuse.
         */
        void reset() {
            size = 0;
        }

        /**
         * Adds a selected neighbor to the cache, copying its vector.
         */
        void add(int source, OnDiskGraphIndex.View view, int node, float score, VectorFloat<?> vec) {
            sourceIdx[size] = source;
            views[size] = view;
            nodes[size] = node;
            scores[size] = score;
            vecs[size].copyFrom(vec, 0, 0, vec.length());
            size++;
        }
    }

}

