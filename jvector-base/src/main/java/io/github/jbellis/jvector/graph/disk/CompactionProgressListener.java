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

/**
 * Callback interface for monitoring the I/O progress of an
 * {@link OnDiskGraphIndexCompactor} run.
 *
 * <p>The compactor processes nodes in batches, writing each batch to disk as it
 * completes. {@link #onProgress} is called every ten batches so that the caller
 * can update a live status display or health-check endpoint without incurring the
 * overhead of a callback on every single batch.
 *
 * <p>For graphs with multiple hierarchical levels the compactor calls
 * {@link #onProgress} separately per level; the {@code totalBatches} value resets
 * to the batch count of the new level at the start of each level. Level&nbsp;0
 * (the base layer) is by far the largest and dominates total runtime, so
 * level-local progress is a good proxy for overall merge progress.
 *
 * <p>Implementations must be thread-safe: callbacks are issued from whichever
 * thread drives the completion service inside
 * {@code OnDiskGraphIndexCompactor.runBatchesWithBackpressure}.
 */
@FunctionalInterface
public interface CompactionProgressListener {

    /**
     * Called periodically as batches are written to disk.
     *
     * @param completedBatches number of batches written to disk so far in the
     *                         current level (always a positive multiple of 10,
     *                         or equal to {@code totalBatches} for the final call)
     * @param totalBatches     total number of batches for the current level
     */
    void onProgress(long completedBatches, long totalBatches);
}
