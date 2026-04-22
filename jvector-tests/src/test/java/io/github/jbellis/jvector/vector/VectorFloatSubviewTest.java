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
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

class VectorFloatSubviewTest
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    void arrayVectorFloatSubviewSharesBackingStorage()
    {
        float[] raw = {10f, 20f, 30f, 40f, 50f};
        ArrayVectorFloat src = new ArrayVectorFloat(raw);
        VectorFloat<?> view = src.subview(1, 3);
        assertInstanceOf(ArraySliceVectorFloat.class, view);
        assertEquals(3, view.length());
        assertEquals(20f, view.get(0), 0f);
        assertEquals(40f, view.get(2), 0f);
        // mutating the source is visible through the view
        raw[1] = 99f;
        assertEquals(99f, view.get(0), 0f);
    }

    @Test
    void bufferVectorFloatSubviewSharesBackingStorage()
    {
        ByteBuffer bb = ByteBuffer.allocate(6 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 6; i++) bb.putFloat(i * 1.5f);
        bb.rewind();
        BufferVectorFloat src = new BufferVectorFloat(bb);
        VectorFloat<?> view = src.subview(2, 3);
        assertInstanceOf(BufferVectorFloat.class, view);
        assertEquals(3, view.length());
        assertEquals(3f, view.get(0), 0f);
        assertEquals(6f, view.get(2), 0f);
        // mutating the source is visible through the view
        bb.putFloat(2 * Float.BYTES, 99f);
        assertEquals(99f, view.get(0), 0f);
    }

    @Test
    void nestedSubviewIsAliasNotCopy()
    {
        float[] raw = new float[20];
        for (int i = 0; i < raw.length; i++) raw[i] = i;
        ArrayVectorFloat src = new ArrayVectorFloat(raw);
        VectorFloat<?> outer = src.subview(5, 10);           // elements 5..14
        VectorFloat<?> inner = outer.subview(2, 3);          // elements 7..9
        assertEquals(3, inner.length());
        assertEquals(7f, inner.get(0), 0f);
        assertEquals(9f, inner.get(2), 0f);
        raw[7] = 777f;
        assertEquals(777f, inner.get(0), 0f);
    }

    @Test
    void fullLengthSubviewReturnsSelf()
    {
        ArrayVectorFloat src = new ArrayVectorFloat(new float[]{1f, 2f, 3f});
        assertSame(src, src.subview(0, 3), "full-range subview should return the source");
    }

    @Test
    void subviewParticipatesInSimdDistanceComputation()
    {
        // Build two full-size vectors; assert that squareL2Distance over (view, view) yields
        // the same result as (materialized subvector, materialized subvector).
        int dim = 64, sub = 16, off = 16;
        float[] a = new float[dim], b = new float[dim];
        for (int i = 0; i < dim; i++) {
            a[i] = (float) Math.sin(i * 0.1);
            b[i] = (float) Math.cos(i * 0.1);
        }
        ArrayVectorFloat av = new ArrayVectorFloat(a);
        ArrayVectorFloat bv = new ArrayVectorFloat(b);

        float viaView = VectorUtil.squareL2Distance(av.subview(off, sub), bv.subview(off, sub));
        float viaCopy = VectorUtil.squareL2Distance(
                VTS.createFloatVector(java.util.Arrays.copyOfRange(a, off, off + sub)),
                VTS.createFloatVector(java.util.Arrays.copyOfRange(b, off, off + sub)));
        assertEquals(viaCopy, viaView, 1e-5f);
    }

    @Test
    void subviewMatchesDotProductAndCosine()
    {
        int dim = 48, off = 8, sub = 24;
        float[] a = new float[dim], b = new float[dim];
        for (int i = 0; i < dim; i++) {
            a[i] = 1f + i * 0.25f;
            b[i] = 2f - i * 0.125f;
        }
        VectorFloat<?> av = VTS.createFloatVector(a);
        VectorFloat<?> bv = VTS.createFloatVector(b);
        VectorFloat<?> aCopy = VTS.createFloatVector(java.util.Arrays.copyOfRange(a, off, off + sub));
        VectorFloat<?> bCopy = VTS.createFloatVector(java.util.Arrays.copyOfRange(b, off, off + sub));

        assertEquals(VectorUtil.dotProduct(aCopy, bCopy),
                     VectorUtil.dotProduct(av.subview(off, sub), bv.subview(off, sub)), 1e-5f);
        assertEquals(VectorUtil.cosine(aCopy, bCopy),
                     VectorUtil.cosine(av.subview(off, sub), bv.subview(off, sub)), 1e-5f);
    }

    @Test
    void rejectsOutOfRangeSubview()
    {
        ArrayVectorFloat src = new ArrayVectorFloat(new float[5]);
        assertThrows(IllegalArgumentException.class, () -> src.subview(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> src.subview(0, 6));
        assertThrows(IllegalArgumentException.class, () -> src.subview(3, 3));
    }
}
