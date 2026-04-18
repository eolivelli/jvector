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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.Buffer;

/**
 * VectorFloat implementation backed by an on-heap MemorySegment.
 */
final public class MemorySegmentVectorFloat implements VectorFloat<MemorySegment>
{
    private final MemorySegment segment;

    MemorySegmentVectorFloat(int length) {
        segment = MemorySegment.ofArray(new float[length]);
    }

    /**
     * @deprecated This constructor <em>copies</em> the buffer contents. To share storage without
     * copying, use {@link #wrap(java.nio.ByteBuffer)} instead.
     */
    @Deprecated
    MemorySegmentVectorFloat(Buffer buffer) {
        this(buffer.remaining());
        segment.copyFrom(MemorySegment.ofBuffer(buffer));
    }

    MemorySegmentVectorFloat(float[] data) {
        this.segment = MemorySegment.ofArray(data);
    }

    private MemorySegmentVectorFloat(MemorySegment segment) {
        this.segment = segment;
    }

    /**
     * Create a view of the given {@link java.nio.ByteBuffer} as a MemorySegment-backed vector
     * without copying. The buffer's bytes are interpreted as native-layout IEEE 754 floats in
     * the buffer's current {@link java.nio.ByteOrder} (jvector's SIMD paths use little-endian).
     *
     * <p>In contrast to {@link #MemorySegmentVectorFloat(Buffer)} which eagerly copies into a
     * fresh segment, this factory wraps the buffer's storage in place via
     * {@link MemorySegment#ofBuffer(java.nio.Buffer)}.
     */
    public static MemorySegmentVectorFloat wrap(java.nio.ByteBuffer buffer) {
        if ((buffer.remaining() % Float.BYTES) != 0) {
            throw new IllegalArgumentException(
                    "ByteBuffer remaining() must be a multiple of Float.BYTES, was " + buffer.remaining());
        }
        if (buffer.order() != java.nio.ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException(
                    "Native SIMD path requires ByteOrder.LITTLE_ENDIAN, got " + buffer.order());
        }
        return new MemorySegmentVectorFloat(MemorySegment.ofBuffer(buffer));
    }

    @Override
    public long ramBytesUsed()
    {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return OH_BYTES + REF_BYTES + segment.byteSize();
    }

    @Override
    public MemorySegment get()
    {
        return segment;
    }

    @Override
    public float get(int n)
    {
        // Fast path for on-heap segments: direct float[] indexing is meaningfully faster than
        // the generic MemorySegment accessors. Fall back to segment access for off-heap buffers
        // (e.g., native memory, direct ByteBuffers) where heapBase() is empty.
        var heap = segment.heapBase();
        if (heap.isPresent()) {
            return ((float[]) heap.get())[n];
        }
        return segment.getAtIndex(ValueLayout.JAVA_FLOAT, n);
    }

    @Override
    public void set(int n, float value)
    {
        var heap = segment.heapBase();
        if (heap.isPresent()) {
            ((float[]) heap.get())[n] = value;
            return;
        }
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, n, value);
    }

    @Override
    public void zero() {
        segment.fill((byte) 0);
    }

    @Override
    public int length() {
        return (int) (segment.byteSize() / Float.BYTES);
    }

    @Override
    public int offset(int i)
    {
        return i * Float.BYTES;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException {
        for (int i = 0; i < length(); i++) {
            writer.writeFloat(get(i));
        }
    }

    @Override
    public VectorFloat<MemorySegment> copy()
    {
        MemorySegmentVectorFloat copy = new MemorySegmentVectorFloat(length());
        copy.copyFrom(this, 0, 0, length());
        return copy;
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        if (src instanceof MemorySegmentVectorFloat) {
            MemorySegmentVectorFloat csrc = (MemorySegmentVectorFloat) src;
            segment.asSlice((long) destOffset * Float.BYTES, (long) length * Float.BYTES)
                    .copyFrom(csrc.segment.asSlice((long) srcOffset * Float.BYTES, (long) length * Float.BYTES));
            return;
        }
        // generic fallback for ArrayVectorFloat, BufferVectorFloat, or any other VectorFloat impl
        for (int i = 0; i < length; i++) {
            set(destOffset + i, src.get(srcOffset + i));
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < Math.min(length(), 25); i++) {
            sb.append(get(i));
            if (i < length() - 1) {
                sb.append(", ");
            }
        }
        if (length() > 25) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MemorySegmentVectorFloat that = (MemorySegmentVectorFloat) o;
        return segment.mismatch(that.segment) == -1;
    }

    @Override
    public int hashCode() {
        return this.getHashCode();
    }
}
