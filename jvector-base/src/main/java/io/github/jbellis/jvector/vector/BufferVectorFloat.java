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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * VectorFloat implementation backed by a caller-owned {@link ByteBuffer}.
 *
 * <p>Storage is not copied on construction: the returned instance is a <em>view</em> over the
 * supplied buffer. This lets callers that already hold vector data as ByteBuffer (for example
 * a database reading rows off disk) feed jvector without the usual {@code float[]} allocation
 * and copy. The buffer's current {@link ByteOrder} is preserved and honored on every element
 * access, so both little- and big-endian layouts work.
 *
 * <p>Element {@code i} is read as {@code data.getFloat(byteOffset + i * Float.BYTES)} where
 * {@code byteOffset} is fixed at construction and does not depend on the buffer's mutable
 * position/limit. This makes the view safe to share across threads that only read it, and
 * robust against callers that continue to position/limit the underlying buffer.
 *
 * <p>The Panama SIMD backend detects instances of this class and dispatches to
 * {@code FloatVector.fromMemorySegment(species, MemorySegment.ofBuffer(data), byteOffset, order)}
 * so SIMD is preserved with no {@code float[]} materialization.
 */
public final class BufferVectorFloat implements VectorFloat<ByteBuffer>
{
    private final ByteBuffer data;
    private final int byteOffset;
    private final int floatLength;

    /**
     * Zero-filled allocation; behaves like {@code new float[length]}. Buffer is native-endian
     * and on-heap.
     */
    public BufferVectorFloat(int length)
    {
        if (length < 0) {
            throw new IllegalArgumentException("length must be >= 0, was " + length);
        }
        this.data = ByteBuffer.allocate(length * Float.BYTES).order(ByteOrder.nativeOrder());
        this.byteOffset = 0;
        this.floatLength = length;
    }

    /**
     * Wrap the entire remaining contents of {@code data} as a float vector view. The buffer's
     * {@code remaining()} must be a multiple of {@code Float.BYTES}.
     */
    public BufferVectorFloat(ByteBuffer data)
    {
        this(data, 0, checkedFloatRemaining(data));
    }

    private static int checkedFloatRemaining(ByteBuffer data)
    {
        Objects.requireNonNull(data, "data");
        if ((data.remaining() % Float.BYTES) != 0) {
            throw new IllegalArgumentException(
                    "ByteBuffer remaining() must be a multiple of Float.BYTES, was " + data.remaining());
        }
        return data.remaining() / Float.BYTES;
    }

    /**
     * Sub-range wrap. {@code floatOffset} and {@code floatLength} are expressed in floats,
     * relative to the buffer's current {@code position()}.
     *
     * <p>The returned view takes an independent {@link ByteBuffer#slice() slice} of the caller's
     * buffer; this lets later mutation of the caller's {@code position}/{@code limit} leave the
     * view intact, and it keeps the base byte offset inside the view stable at zero so that
     * {@code MemorySegment.ofBuffer(view)} can be called without further per-access adjustment
     * in the Panama SIMD path.
     */
    public BufferVectorFloat(ByteBuffer data, int floatOffset, int floatLength)
    {
        Objects.requireNonNull(data, "data");
        if (floatOffset < 0 || floatLength < 0) {
            throw new IllegalArgumentException("offset/length must be >= 0");
        }
        long byteEnd = (long) floatOffset * Float.BYTES + (long) floatLength * Float.BYTES;
        if (byteEnd > data.remaining()) {
            throw new IllegalArgumentException(
                    "view [" + floatOffset + "," + (floatOffset + floatLength)
                    + ") floats exceeds buffer remaining=" + data.remaining() + " bytes");
        }
        ByteBuffer dup = data.duplicate().order(data.order());
        int startByte = data.position() + floatOffset * Float.BYTES;
        int endByte = startByte + floatLength * Float.BYTES;
        dup.position(startByte).limit(endByte);
        this.data = dup.slice().order(data.order());
        this.byteOffset = 0;
        this.floatLength = floatLength;
    }

    /**
     * @return the backing buffer. Mutating its position/limit does not affect element access,
     * but mutating its contents does.
     */
    @Override
    public ByteBuffer get()
    {
        return data;
    }

    /** @return byte offset at which this view's element 0 lives within the backing buffer. */
    public int byteOffset()
    {
        return byteOffset;
    }

    /** @return the byte order the backing buffer is read with. */
    public ByteOrder byteOrder()
    {
        return data.order();
    }

    @Override
    public float get(int i)
    {
        return data.getFloat(byteOffset + i * Float.BYTES);
    }

    @Override
    public void set(int i, float value)
    {
        data.putFloat(byteOffset + i * Float.BYTES, value);
    }

    @Override
    public void zero()
    {
        for (int i = 0; i < floatLength; i++) {
            data.putFloat(byteOffset + i * Float.BYTES, 0f);
        }
    }

    @Override
    public int length()
    {
        return floatLength;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException
    {
        for (int i = 0; i < floatLength; i++) {
            writer.writeFloat(get(i));
        }
    }

    @Override
    public VectorFloat<ByteBuffer> copy()
    {
        ByteBuffer owned = ByteBuffer.allocate(floatLength * Float.BYTES).order(data.order());
        ByteBuffer srcView = data.duplicate().order(data.order());
        srcView.position(byteOffset).limit(byteOffset + floatLength * Float.BYTES);
        owned.put(srcView);
        owned.rewind();
        return new BufferVectorFloat(owned, 0, floatLength);
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        if (src instanceof BufferVectorFloat) {
            BufferVectorFloat bsrc = (BufferVectorFloat) src;
            if (bsrc.byteOrder() == this.byteOrder()) {
                ByteBuffer srcDup = bsrc.data.duplicate().order(bsrc.data.order());
                int srcStart = bsrc.byteOffset + srcOffset * Float.BYTES;
                int bytes = length * Float.BYTES;
                srcDup.position(srcStart).limit(srcStart + bytes);
                ByteBuffer destDup = this.data.duplicate().order(this.data.order());
                destDup.position(this.byteOffset + destOffset * Float.BYTES);
                destDup.put(srcDup);
                return;
            }
        } else if (src instanceof ArrayVectorFloat) {
            float[] a = ((ArrayVectorFloat) src).get();
            for (int i = 0; i < length; i++) {
                set(destOffset + i, a[srcOffset + i]);
            }
            return;
        }
        for (int i = 0; i < length; i++) {
            set(destOffset + i, src.get(srcOffset + i));
        }
    }

    @Override
    public long ramBytesUsed()
    {
        long OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return OH_BYTES + REF_BYTES + 2L * Integer.BYTES + (long) floatLength * Float.BYTES;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int preview = Math.min(floatLength, 25);
        for (int i = 0; i < preview; i++) {
            sb.append(get(i));
            if (i < floatLength - 1) {
                sb.append(", ");
            }
        }
        if (floatLength > 25) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (!(o instanceof VectorFloat<?>)) return false;
        VectorFloat<?> that = (VectorFloat<?>) o;
        if (that.length() != this.floatLength) return false;
        for (int i = 0; i < floatLength; i++) {
            if (Float.floatToIntBits(get(i)) != Float.floatToIntBits(that.get(i))) return false;
        }
        return true;
    }

    @Override
    public int hashCode()
    {
        return this.getHashCode();
    }
}
