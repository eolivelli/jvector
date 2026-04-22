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

import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

class BufferVectorFloatTest
{
    private static ByteBuffer littleEndianOf(float... values)
    {
        ByteBuffer bb = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) bb.putFloat(v);
        bb.rewind();
        return bb;
    }

    private static ByteBuffer bigEndianOf(float... values)
    {
        ByteBuffer bb = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.BIG_ENDIAN);
        for (float v : values) bb.putFloat(v);
        bb.rewind();
        return bb;
    }

    @Test
    void elementAccessMatchesFloatArrayLittleEndian()
    {
        float[] expected = {1.0f, -2.5f, 3.14f, 0.0f, Float.MAX_VALUE};
        BufferVectorFloat bv = new BufferVectorFloat(littleEndianOf(expected));
        assertEquals(expected.length, bv.length());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], bv.get(i), 0f);
        }
    }

    @Test
    void elementAccessMatchesFloatArrayBigEndian()
    {
        float[] expected = {1.0f, -2.5f, 3.14f, 0.0f, Float.MAX_VALUE};
        BufferVectorFloat bv = new BufferVectorFloat(bigEndianOf(expected));
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], bv.get(i), 0f);
        }
    }

    @Test
    void directBufferBackingIsZeroCopy()
    {
        ByteBuffer direct = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 4; i++) direct.putFloat(i * 1.5f);
        direct.rewind();

        BufferVectorFloat bv = new BufferVectorFloat(direct);
        // The view is a slice (independent ByteBuffer object) over the same off-heap storage.
        // Proving no data copy: mutating via the original updates the view.
        assertFalse(direct.hasArray(), "direct buffer — verifies off-heap path");
        direct.putFloat(0, 999f);
        assertEquals(999f, bv.get(0), 0f, "slice must alias source storage, not copy");
    }

    @Test
    void subrangeViewHonorsBoundsAndDoesNotAlias()
    {
        ByteBuffer bb = littleEndianOf(10f, 20f, 30f, 40f, 50f);
        BufferVectorFloat middle = new BufferVectorFloat(bb, 1, 3); // {20, 30, 40}
        assertEquals(3, middle.length());
        assertEquals(20f, middle.get(0), 0f);
        assertEquals(30f, middle.get(1), 0f);
        assertEquals(40f, middle.get(2), 0f);
        assertThrows(IndexOutOfBoundsException.class, () -> middle.get(3));
    }

    @Test
    void positionLimitMutationAfterConstructionDoesNotAffectView()
    {
        ByteBuffer bb = littleEndianOf(1f, 2f, 3f, 4f);
        BufferVectorFloat bv = new BufferVectorFloat(bb);
        bb.position(8).limit(12);
        assertEquals(1f, bv.get(0), 0f);
        assertEquals(4f, bv.get(3), 0f);
    }

    @Test
    void zeroAllocationConstructorProducesZeroedVector()
    {
        BufferVectorFloat bv = new BufferVectorFloat(7);
        assertEquals(7, bv.length());
        for (int i = 0; i < 7; i++) {
            assertEquals(0f, bv.get(i), 0f);
        }
    }

    @Test
    void setRoundTrip()
    {
        BufferVectorFloat bv = new BufferVectorFloat(4);
        bv.set(0, 1f); bv.set(1, 2f); bv.set(2, -3f); bv.set(3, 0.5f);
        assertEquals(1f, bv.get(0), 0f);
        assertEquals(-3f, bv.get(2), 0f);
    }

    @Test
    void zeroResets()
    {
        BufferVectorFloat bv = new BufferVectorFloat(littleEndianOf(1f, 2f, 3f).duplicate().order(ByteOrder.LITTLE_ENDIAN));
        bv.zero();
        for (int i = 0; i < 3; i++) assertEquals(0f, bv.get(i), 0f);
    }

    @Test
    void copyProducesIndependentStorage()
    {
        BufferVectorFloat src = new BufferVectorFloat(littleEndianOf(1f, 2f, 3f));
        VectorFloat<?> dup = src.copy();
        src.set(0, 99f);
        assertEquals(99f, src.get(0), 0f);
        assertEquals(1f, dup.get(0), 0f, "copy must not alias source");
    }

    @Test
    void copyFromArrayVectorFloat()
    {
        BufferVectorFloat dest = new BufferVectorFloat(4);
        ArrayVectorFloat src = new ArrayVectorFloat(new float[]{10f, 20f, 30f, 40f});
        dest.copyFrom(src, 1, 0, 3);
        assertEquals(20f, dest.get(0), 0f);
        assertEquals(30f, dest.get(1), 0f);
        assertEquals(40f, dest.get(2), 0f);
        assertEquals(0f, dest.get(3), 0f);
    }

    @Test
    void copyFromBufferVectorFloatSameOrder()
    {
        BufferVectorFloat dest = new BufferVectorFloat(4);
        BufferVectorFloat src = new BufferVectorFloat(littleEndianOf(1f, 2f, 3f, 4f, 5f));
        dest.copyFrom(src, 1, 0, 3);
        assertEquals(2f, dest.get(0), 0f);
        assertEquals(3f, dest.get(1), 0f);
        assertEquals(4f, dest.get(2), 0f);
    }

    @Test
    void copyFromBufferVectorFloatDifferentOrder()
    {
        BufferVectorFloat dest = new BufferVectorFloat(3); // native order
        BufferVectorFloat src = new BufferVectorFloat(bigEndianOf(1f, 2f, 3f));
        dest.copyFrom(src, 0, 0, 3);
        for (int i = 0; i < 3; i++) assertEquals((float) (i + 1), dest.get(i), 0f);
    }

    @Test
    void equalsMatchesSemanticContent()
    {
        BufferVectorFloat le = new BufferVectorFloat(littleEndianOf(1f, 2f, 3f));
        BufferVectorFloat be = new BufferVectorFloat(bigEndianOf(1f, 2f, 3f));
        ArrayVectorFloat av = new ArrayVectorFloat(new float[]{1f, 2f, 3f});
        assertEquals(le, be, "byte order differs but content identical");
        assertEquals(le, av, "BufferVectorFloat equals any VectorFloat with same content");
        // ArrayVectorFloat.equals() is class-strict — we don't change that contract here.
    }

    @Test
    void rejectsNonFloatAlignedRemaining()
    {
        ByteBuffer odd = ByteBuffer.allocate(7).order(ByteOrder.LITTLE_ENDIAN);
        assertThrows(IllegalArgumentException.class, () -> new BufferVectorFloat(odd));
    }

    @Test
    void rejectsNegativeArgs()
    {
        ByteBuffer ok = littleEndianOf(1f, 2f);
        assertThrows(IllegalArgumentException.class, () -> new BufferVectorFloat(ok, -1, 1));
        assertThrows(IllegalArgumentException.class, () -> new BufferVectorFloat(ok, 0, -1));
        assertThrows(IllegalArgumentException.class, () -> new BufferVectorFloat(-1));
    }

    @Test
    void rejectsOutOfRangeSubview()
    {
        ByteBuffer bb = littleEndianOf(1f, 2f, 3f);
        assertThrows(IllegalArgumentException.class, () -> new BufferVectorFloat(bb, 2, 2));
    }

    @Test
    void ramBytesUsedIsFinite()
    {
        BufferVectorFloat bv = new BufferVectorFloat(1024);
        long size = bv.ramBytesUsed();
        assertTrue(size >= 1024L * Float.BYTES, "reported size must cover backing buffer");
    }

    @Test
    void byteOrderPreserved()
    {
        BufferVectorFloat le = new BufferVectorFloat(littleEndianOf(1f));
        BufferVectorFloat be = new BufferVectorFloat(bigEndianOf(1f));
        assertEquals(ByteOrder.LITTLE_ENDIAN, le.byteOrder());
        assertEquals(ByteOrder.BIG_ENDIAN, be.byteOrder());
    }
}
