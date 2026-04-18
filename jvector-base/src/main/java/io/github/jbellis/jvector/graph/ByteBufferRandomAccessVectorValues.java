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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * A {@link RandomAccessVectorValues} backed by a single caller-owned {@link ByteBuffer} that
 * holds {@code count × dimension × Float.BYTES} bytes of concatenated IEEE 754 floats.
 *
 * <p>Vectors are never copied: {@link #getVector(int)} returns a view (via the active
 * {@link VectorTypeSupport#wrapFloatVector(ByteBuffer, int, int)}) that aliases the source
 * bytes. This lets integrators whose data is already in ByteBuffer form (for example a
 * database page cache) feed {@code GraphIndexBuilder} without the usual {@code float[]}
 * materialization step.
 *
 * <p>The buffer is pre-sliced at construction time so later mutation of the caller's buffer
 * position / limit does not disturb this RAVV. Mutating the buffer's <em>contents</em> after
 * construction is visible to this RAVV — callers are responsible for ensuring writers and
 * readers do not race.
 */
public class ByteBufferRandomAccessVectorValues implements RandomAccessVectorValues
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private final ByteBuffer data;
    private final int count;
    private final int dimension;

    public ByteBufferRandomAccessVectorValues(ByteBuffer data, int count, int dimension)
    {
        Objects.requireNonNull(data, "data");
        if (count < 0) throw new IllegalArgumentException("count must be >= 0, was " + count);
        if (dimension <= 0) throw new IllegalArgumentException("dimension must be > 0, was " + dimension);
        long need = (long) count * dimension * Float.BYTES;
        if (data.remaining() < need) {
            throw new IllegalArgumentException(
                    "buffer too small: need " + need + " bytes, have " + data.remaining());
        }
        ByteBuffer dup = data.duplicate().order(data.order());
        int startByte = data.position();
        dup.position(startByte).limit(startByte + (int) need);
        this.data = dup.slice().order(data.order());
        this.count = count;
        this.dimension = dimension;
    }

    @Override
    public int size()
    {
        return count;
    }

    @Override
    public int dimension()
    {
        return dimension;
    }

    @Override
    public VectorFloat<?> getVector(int targetOrd)
    {
        if (targetOrd < 0 || targetOrd >= count) {
            throw new IndexOutOfBoundsException("ordinal " + targetOrd + " out of [0, " + count + ")");
        }
        return VTS.wrapFloatVector(data, targetOrd * dimension, dimension);
    }

    @Override
    public boolean isValueShared()
    {
        // Each getVector returns a distinct view instance — callers may hold onto results.
        return false;
    }

    @Override
    public ByteBufferRandomAccessVectorValues copy()
    {
        return this;
    }

    /**
     * Convenience factory accepting a plain {@code float[]} (copying it into a native-endian
     * ByteBuffer). Primarily used by tests that already have {@code float[]} fixtures.
     */
    public static ByteBufferRandomAccessVectorValues fromFloats(float[][] vectors, int dimension)
    {
        Objects.requireNonNull(vectors, "vectors");
        if (dimension <= 0) throw new IllegalArgumentException("dimension > 0 required");
        int count = vectors.length;
        ByteBuffer bb = ByteBuffer.allocate(count * dimension * Float.BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float[] v : vectors) {
            if (v.length != dimension) {
                throw new IllegalArgumentException("vector dimension mismatch: " + v.length + " vs " + dimension);
            }
            for (float f : v) bb.putFloat(f);
        }
        bb.rewind();
        return new ByteBufferRandomAccessVectorValues(bb, count, dimension);
    }

    /** @return the live view over the backing data (for tests/integration). */
    public ByteBuffer data()
    {
        return data;
    }
}
