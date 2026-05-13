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

package io.github.jbellis.jvector.disk;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;

/**
 * This is a subset of DataInput, plus seek and readFully methods, which allows implementations
 * to use more efficient options like FloatBuffer for bulk reads.
 * <p>
 * JVector includes production-ready implementations; the recommended way to use these are via
 * `ReaderSupplierFactory.open`.  For custom implementations, e.g. reading from network storage,
 * you should also implement a corresponding `ReaderSupplier`.
 * <p>
 * The general usage pattern is expected to be "seek to a position, then read sequentially from there."
 * Thus, RandomAccessReader implementations are expected to be stateful and NOT threadsafe; JVector
 * uses the ReaderSupplier API to create a RandomAccessReader per thread, as needed.
 */
public interface RandomAccessReader extends AutoCloseable {
    /**
     * Seeks to the specified offset.
     * @param offset the offset to seek to
     * @throws IOException if an I/O error occurs
     */
    void seek(long offset) throws IOException;

    /**
     * Returns the current position.
     * @return the current position
     * @throws IOException if an I/O error occurs
     */
    long getPosition() throws IOException;

    /**
     * Reads an integer.
     * @return the integer value
     * @throws IOException if an I/O error occurs
     */
    int readInt() throws IOException;

    /**
     * Reads a float.
     * @return the float value
     * @throws IOException if an I/O error occurs
     */
    float readFloat() throws IOException;

    /**
     * Reads a long.
     * @return the long value
     * @throws IOException if an I/O error occurs
     */
    long readLong() throws IOException;

    /**
     * Reads bytes into the array.
     * @param bytes the byte array to read into
     * @throws IOException if an I/O error occurs
     */
    void readFully(byte[] bytes) throws IOException;

    /**
     * Reads bytes into the buffer.
     * @param buffer the ByteBuffer to read into
     * @throws IOException if an I/O error occurs
     */
    void readFully(ByteBuffer buffer) throws IOException;

    /**
     * Reads floats into the array.
     * @param floats the float array to read into
     * @throws IOException if an I/O error occurs
     */
    default void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    /**
     * Reads longs into the array.
     * @param vector the long array to read into
     * @throws IOException if an I/O error occurs
     */
    void readFully(long[] vector) throws IOException;

    /**
     * Reads integers into the array.
     * @param ints the int array to read into
     * @param offset the offset in the array
     * @param count the number of integers to read
     * @throws IOException if an I/O error occurs
     */
    void read(int[] ints, int offset, int count) throws IOException;

    /**
     * Reads floats into the array.
     * @param floats the float array to read into
     * @param offset the offset in the array
     * @param count the number of floats to read
     * @throws IOException if an I/O error occurs
     */
    void read(float[] floats, int offset, int count) throws IOException;

    /**
     * Closes this reader.
     * @throws IOException if an I/O error occurs
     */
    void close() throws IOException;

    /**
     * Returns the length of the reader slice.
     * @return the length
     * @throws IOException if an I/O error occurs
     */
    long length() throws IOException;

    /**
     * Asynchronously read {@code length} bytes starting at {@code offset}. The returned future
     * completes with a ByteBuffer positioned at 0 and limited to {@code length}.
     *
     * <p>Contract:
     * <ul>
     *   <li>Must NOT modify this reader's current seek position.</li>
     *   <li>Must NOT block the caller waiting for IO to complete. Implementations backed by a
     *       parallel-capable backend (e.g. a network client) should dispatch the read and return
     *       the future immediately.</li>
     *   <li>Multiple async reads may be in flight concurrently. The async path is logically
     *       routed through a shared backend that bypasses this reader instance's position
     *       cursor.</li>
     * </ul>
     *
     * <p>The default implementation is a synchronous fallback that saves and restores the current
     * position around a {@code seek}/{@code readFully} pair, returning a completed future. This
     * preserves correctness for local file/mmap readers without giving them any actual concurrency
     * benefit; readers that wrap a non-blocking backend should override.
     *
     * @param offset starting offset
     * @param length number of bytes to read
     * @return a future completing with the read bytes
     */
    default CompletableFuture<ByteBuffer> readRangeAsync(long offset, int length) {
        try {
            long saved = getPosition();
            byte[] bytes = new byte[length];
            seek(offset);
            readFully(bytes);
            seek(saved);
            return CompletableFuture.completedFuture(ByteBuffer.wrap(bytes));
        } catch (IOException e) {
            CompletableFuture<ByteBuffer> failed = new CompletableFuture<>();
            failed.completeExceptionally(e);
            return failed;
        }
    }
}
