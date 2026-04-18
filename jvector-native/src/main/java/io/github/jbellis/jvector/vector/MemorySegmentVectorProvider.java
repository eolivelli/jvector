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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;

/**
 * VectorTypeSupport using MemorySegments.
 */
public class MemorySegmentVectorProvider implements VectorTypeSupport
{
    @Override
    public VectorFloat<?> createFloatVector(Object data)
    {
        if (data instanceof Buffer)
            return new MemorySegmentVectorFloat((Buffer) data);

        return new MemorySegmentVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatVector(int length)
    {
        return new MemorySegmentVectorFloat(length);
    }

    /**
     * Zero-copy wrap that returns a MemorySegment-backed view when the buffer's layout matches
     * the native SIMD contract (little-endian), so {@code FloatVector.fromMemorySegment} can
     * run on it directly. Big-endian buffers fall through to the base {@link BufferVectorFloat}
     * view — still zero-copy but dispatched through the Panama polymorphic path.
     */
    @Override
    public VectorFloat<?> wrapFloatVector(ByteBuffer data, int floatOffset, int floatLength)
    {
        if (data.order() != java.nio.ByteOrder.LITTLE_ENDIAN) {
            return VectorTypeSupport.super.wrapFloatVector(data, floatOffset, floatLength);
        }
        ByteBuffer dup = data.duplicate().order(data.order());
        int startByte = data.position() + floatOffset * Float.BYTES;
        dup.position(startByte).limit(startByte + floatLength * Float.BYTES);
        return MemorySegmentVectorFloat.wrap(dup);
    }

    @Override
    public VectorFloat<?> readFloatVector(RandomAccessReader r, int size) throws IOException
    {
        float[] data = new float[size];
        r.readFully(data);
        return new MemorySegmentVectorFloat(data);
    }

    @Override
    public void readFloatVector(RandomAccessReader r, int count, VectorFloat<?> vector, int offset) throws IOException {
        float[] dest = (float[]) ((MemorySegmentVectorFloat) vector).get().heapBase().get();
        r.read(dest, offset, count);
    }

    @Override
    public void writeFloatVector(IndexWriter out, VectorFloat<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeFloat(vector.get(i));
    }

    @Override
    public ByteSequence<?> createByteSequence(Object data)
    {
        if (data instanceof Buffer)
            return new MemorySegmentByteSequence((Buffer) data);

        return new MemorySegmentByteSequence((byte[]) data);
    }

    @Override
    public ByteSequence<?> createByteSequence(int length)
    {
        return new MemorySegmentByteSequence(length);
    }

    @Override
    public ByteSequence<?> readByteSequence(RandomAccessReader r, int size) throws IOException
    {
        var vector = new MemorySegmentByteSequence(size);
        r.readFully(vector.get().asByteBuffer());
        return vector;
    }

    @Override
    public void readByteSequence(RandomAccessReader r, ByteSequence<?> sequence) throws IOException {
        r.readFully(((MemorySegmentByteSequence) sequence).get().asByteBuffer());
    }


    @Override
    public void writeByteSequence(IndexWriter out, ByteSequence<?> sequence) throws IOException
    {
        for (int i = 0; i < sequence.length(); i++)
            out.writeByte(sequence.get(i));
    }
}
