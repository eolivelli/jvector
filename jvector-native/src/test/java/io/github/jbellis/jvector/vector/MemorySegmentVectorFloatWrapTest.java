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

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

class MemorySegmentVectorFloatWrapTest
{
    @Test
    void wrapAliasesBufferStorage()
    {
        ByteBuffer bb = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 4; i++) bb.putFloat(i * 2f);
        bb.rewind();

        MemorySegmentVectorFloat v = MemorySegmentVectorFloat.wrap(bb);
        assertEquals(4, v.length());
        assertEquals(0f, v.get(0), 0f);
        assertEquals(6f, v.get(3), 0f);

        // mutate through the original ByteBuffer, observe the change through the view
        bb.putFloat(0, 99f);
        assertEquals(99f, v.get(0), 0f, "wrap must not copy — shared storage");
    }

    @Test
    void wrapRejectsBigEndian()
    {
        ByteBuffer be = ByteBuffer.allocate(4 * Float.BYTES).order(ByteOrder.BIG_ENDIAN);
        assertThrows(IllegalArgumentException.class, () -> MemorySegmentVectorFloat.wrap(be));
    }

    @Test
    void wrapRejectsNonFloatAlignedRemaining()
    {
        ByteBuffer odd = ByteBuffer.allocate(7).order(ByteOrder.LITTLE_ENDIAN);
        assertThrows(IllegalArgumentException.class, () -> MemorySegmentVectorFloat.wrap(odd));
    }

    @Test
    void legacyCopyingConstructorStillCopies()
    {
        ByteBuffer bb = ByteBuffer.allocate(4 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 4; i++) bb.putFloat(i * 2f);
        bb.rewind();

        @SuppressWarnings("deprecation")
        MemorySegmentVectorFloat v = new MemorySegmentVectorFloat(bb);
        // Mutating bb must NOT change the copy — proving the semantics of the old ctor.
        bb.putFloat(0, 999f);
        assertEquals(0f, v.get(0), 0f, "copying constructor must not alias");
    }
}
