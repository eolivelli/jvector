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

class VectorTypeSupportByteBufferTest
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final VectorTypeSupport DEFAULT_VTS =
            new DefaultVectorizationProvider().getVectorTypeSupport();

    private static ByteBuffer le(float... floats)
    {
        ByteBuffer bb = ByteBuffer.allocate(floats.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : floats) bb.putFloat(v);
        bb.rewind();
        return bb;
    }

    @Test
    void wrapReturnsReadableVectorAcrossActiveProvider()
    {
        ByteBuffer bb = le(1f, 2f, 3f, 4f);
        VectorFloat<?> v = VTS.wrapFloatVector(bb);
        assertEquals(4, v.length());
        assertEquals(1f, v.get(0), 0f);
        assertEquals(4f, v.get(3), 0f);
    }

    @Test
    void wrapAcrossDefaultProvider()
    {
        ByteBuffer bb = le(1f, 2f, 3f, 4f);
        VectorFloat<?> v = DEFAULT_VTS.wrapFloatVector(bb);
        assertEquals(4, v.length());
        for (int i = 0; i < 4; i++) {
            assertEquals((float) (i + 1), v.get(i), 0f);
        }
    }

    @Test
    void wrapZeroCopyOnDirectBuffer()
    {
        ByteBuffer direct = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 4; i++) direct.putFloat(i * 1.25f);
        direct.rewind();
        VectorFloat<?> v = DEFAULT_VTS.wrapFloatVector(direct);
        // Mutate the source buffer; the view must see the change (proving no copy happened).
        direct.putFloat(0, 999f);
        assertEquals(999f, v.get(0), 0f);
    }

    @Test
    void wrapWithSubrange()
    {
        ByteBuffer bb = le(10f, 20f, 30f, 40f, 50f);
        VectorFloat<?> v = DEFAULT_VTS.wrapFloatVector(bb, 1, 3);
        assertEquals(3, v.length());
        assertEquals(20f, v.get(0), 0f);
        assertEquals(40f, v.get(2), 0f);
    }

    @Test
    void wrapRejectsBadAlignment()
    {
        ByteBuffer odd = ByteBuffer.allocate(7).order(ByteOrder.LITTLE_ENDIAN);
        assertThrows(IllegalArgumentException.class, () -> VTS.wrapFloatVector(odd));
    }

    @Test
    void wrapEquivalentAcrossProviders()
    {
        float[] raw = {-0.5f, 1.5f, 2.75f, -3.125f, 4.5f};
        ByteBuffer bb = le(raw);

        VectorFloat<?> defaultView = DEFAULT_VTS.wrapFloatVector(bb.duplicate().order(ByteOrder.LITTLE_ENDIAN));
        VectorFloat<?> activeView = VTS.wrapFloatVector(bb.duplicate().order(ByteOrder.LITTLE_ENDIAN));

        assertEquals(defaultView.length(), activeView.length());
        for (int i = 0; i < raw.length; i++) {
            assertEquals(defaultView.get(i), activeView.get(i), 0f);
        }
    }
}
