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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.graph.ByteBufferRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;
import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * RAVV backed by a memory-mapped file of contiguous little-endian IEEE 754 floats.
 *
 * <p>Unlike the previous implementation — which copied each row into an on-heap
 * {@code float[dimension]} scratch buffer on every {@link #getVector} call — this version
 * delegates to {@link ByteBufferRandomAccessVectorValues} over the {@link MappedByteBuffer},
 * so reads are served directly from the mmap region with no per-call {@code float[]}
 * allocation.
 */
public class MMapRandomAccessVectorValues implements RandomAccessVectorValues, Closeable
{
    private final int dimension;
    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final ByteBufferRandomAccessVectorValues delegate;

    public MMapRandomAccessVectorValues(File f, int dimension)
    {
        assert f != null && f.exists() && f.canRead();
        long bytesPerVector = (long) dimension * Float.BYTES;
        assert f.length() % bytesPerVector == 0;

        try {
            this.file = f;
            this.dimension = dimension;
            this.raf = new RandomAccessFile(f, "r");
            this.channel = raf.getChannel();
            long size = f.length();
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
            mapped.order(ByteOrder.LITTLE_ENDIAN);
            int count = (int) (size / bytesPerVector);
            this.delegate = new ByteBufferRandomAccessVectorValues(mapped, count, dimension);
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    @Override
    public int size()
    {
        return delegate.size();
    }

    @Override
    public int dimension()
    {
        return dimension;
    }

    @Override
    public VectorFloat<?> getVector(int targetOrd)
    {
        return delegate.getVector(targetOrd);
    }

    @Override
    public boolean isValueShared()
    {
        return delegate.isValueShared();
    }

    @Override
    public RandomAccessVectorValues copy()
    {
        return new MMapRandomAccessVectorValues(file, dimension);
    }

    @Override
    public void close()
    {
        try {
            channel.close();
            raf.close();
        } catch (IOException e) {
            throw new IOError(e);
        }
    }
}
