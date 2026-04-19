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
import java.util.Arrays;

/**
 * A {@link VectorFloat} that is a <em>view</em> into a contiguous range of a root
 * {@code float[]}. Constructing this class does not allocate or copy the backing storage — both
 * element access and SIMD dispatch read the underlying array at {@code arrayOffset + i}.
 *
 * <p>Primarily used inside jvector's Product Quantization training: extracting the M subvectors
 * of each training vector can return M views instead of M materialized {@code float[dim/M]}
 * copies.
 *
 * <p>The companion to {@link ArraySliceByteSequence} for the float-vector side of the type system.
 */
public final class ArraySliceVectorFloat implements VectorFloat<float[]>
{
    private final float[] data;
    private final int arrayOffset;
    private final int length;

    public ArraySliceVectorFloat(float[] data, int arrayOffset, int length)
    {
        if (data == null) throw new NullPointerException("data");
        if (arrayOffset < 0 || length < 0 || (long) arrayOffset + length > data.length) {
            throw new IllegalArgumentException(
                    "slice [" + arrayOffset + "," + (arrayOffset + length)
                            + ") out of range for float[" + data.length + "]");
        }
        this.data = data;
        this.arrayOffset = arrayOffset;
        this.length = length;
    }

    /** @return the root float[] this view borrows from (offsets are still relative to arrayOffset()). */
    @Override
    public float[] get()
    {
        return data;
    }

    /** @return the index into {@link #get()} where this view's element 0 lives. */
    public int arrayOffset()
    {
        return arrayOffset;
    }

    @Override
    public float get(int n)
    {
        return data[arrayOffset + n];
    }

    @Override
    public void set(int n, float value)
    {
        data[arrayOffset + n] = value;
    }

    @Override
    public void zero()
    {
        Arrays.fill(data, arrayOffset, arrayOffset + length, 0f);
    }

    @Override
    public int length()
    {
        return length;
    }

    @Override
    public void writeTo(IndexWriter writer) throws IOException
    {
        writer.writeFloats(data, arrayOffset, length);
    }

    @Override
    public VectorFloat<float[]> copy()
    {
        return new ArrayVectorFloat(Arrays.copyOfRange(data, arrayOffset, arrayOffset + length));
    }

    @Override
    public VectorFloat<?> subview(int floatOffset, int floatLength)
    {
        if (floatOffset < 0 || floatLength < 0 || (long) floatOffset + floatLength > this.length) {
            throw new IllegalArgumentException(
                    "subview [" + floatOffset + "," + (floatOffset + floatLength)
                            + ") out of range for view length " + this.length);
        }
        if (floatOffset == 0 && floatLength == this.length) {
            return this;
        }
        return new ArraySliceVectorFloat(data, arrayOffset + floatOffset, floatLength);
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int copyLength)
    {
        if (src instanceof ArrayVectorFloat csrc) {
            System.arraycopy(csrc.get(), srcOffset, data, arrayOffset + destOffset, copyLength);
            return;
        }
        if (src instanceof ArraySliceVectorFloat asrc) {
            System.arraycopy(asrc.data, asrc.arrayOffset + srcOffset, data, arrayOffset + destOffset, copyLength);
            return;
        }
        for (int i = 0; i < copyLength; i++) {
            data[arrayOffset + destOffset + i] = src.get(srcOffset + i);
        }
    }

    @Override
    public long ramBytesUsed()
    {
        // Only the slice metadata; the backing float[] is accounted by its owner.
        return RamUsageEstimator.NUM_BYTES_OBJECT_HEADER
                + RamUsageEstimator.NUM_BYTES_OBJECT_REF
                + 2L * Integer.BYTES;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int preview = Math.min(length, 25);
        for (int i = 0; i < preview; i++) {
            sb.append(get(i));
            if (i < length - 1) {
                sb.append(", ");
            }
        }
        if (length > 25) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (!(o instanceof VectorFloat<?> that)) return false;
        if (that.length() != this.length) return false;
        for (int i = 0; i < length; i++) {
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
