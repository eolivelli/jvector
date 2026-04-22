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
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ByteBufferRandomAccessVectorValuesTest
{
    private static final VectorTypeSupport VTS =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    private static float[][] randomData(long seed, int count, int dimension)
    {
        Random r = new Random(seed);
        float[][] data = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) {
                data[i][j] = r.nextFloat() * 2f - 1f;
            }
        }
        return data;
    }

    private static ByteBuffer toByteBuffer(float[][] data, int dimension)
    {
        ByteBuffer bb = ByteBuffer.allocate(data.length * dimension * Float.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float[] v : data) for (float f : v) bb.putFloat(f);
        bb.rewind();
        return bb;
    }

    @Test
    void matchesListRAVVElementByElement()
    {
        int count = 100, dim = 32;
        float[][] data = randomData(42, count, dim);
        ByteBuffer bb = toByteBuffer(data, dim);

        ByteBufferRandomAccessVectorValues bbravv = new ByteBufferRandomAccessVectorValues(bb, count, dim);
        List<VectorFloat<?>> list = new ArrayList<>();
        for (float[] v : data) list.add(VTS.createFloatVector(v));
        ListRandomAccessVectorValues listRavv = new ListRandomAccessVectorValues(list, dim);

        assertEquals(count, bbravv.size());
        assertEquals(dim, bbravv.dimension());

        for (int ord = 0; ord < count; ord++) {
            VectorFloat<?> a = bbravv.getVector(ord);
            VectorFloat<?> b = listRavv.getVector(ord);
            assertEquals(b.length(), a.length());
            for (int i = 0; i < dim; i++) {
                assertEquals(b.get(i), a.get(i), 0f, "mismatch at ord=" + ord + " i=" + i);
            }
        }
    }

    @Test
    void rejectsOutOfRangeOrdinal()
    {
        ByteBuffer bb = ByteBuffer.allocate(10 * 4 * 4).order(ByteOrder.LITTLE_ENDIAN);
        ByteBufferRandomAccessVectorValues ravv = new ByteBufferRandomAccessVectorValues(bb, 10, 4);
        assertThrows(IndexOutOfBoundsException.class, () -> ravv.getVector(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> ravv.getVector(10));
    }

    @Test
    void rejectsUndersizedBuffer()
    {
        ByteBuffer tooSmall = ByteBuffer.allocate(10).order(ByteOrder.LITTLE_ENDIAN);
        assertThrows(IllegalArgumentException.class,
                () -> new ByteBufferRandomAccessVectorValues(tooSmall, 100, 4));
    }

    @Test
    void survivesCallerBufferPositionMutation()
    {
        int count = 5, dim = 4;
        float[][] data = randomData(7, count, dim);
        ByteBuffer bb = toByteBuffer(data, dim);

        ByteBufferRandomAccessVectorValues ravv = new ByteBufferRandomAccessVectorValues(bb, count, dim);
        // Caller fiddles with the source buffer after construction — must not disturb the view.
        bb.position(bb.capacity()).limit(bb.capacity());
        VectorFloat<?> v = ravv.getVector(2);
        for (int i = 0; i < dim; i++) {
            assertEquals(data[2][i], v.get(i), 0f);
        }
    }

    @Test
    void threadLocalSupplierGivesIndependentWorkers()
        throws InterruptedException
    {
        int count = 50, dim = 16;
        float[][] data = randomData(11, count, dim);
        ByteBuffer bb = toByteBuffer(data, dim);

        ByteBufferRandomAccessVectorValues ravv = new ByteBufferRandomAccessVectorValues(bb, count, dim);
        ExecutorService exec = Executors.newFixedThreadPool(4);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger mismatches = new AtomicInteger();
        for (int t = 0; t < 4; t++) {
            exec.submit(() -> {
                try {
                    start.await();
                    RandomAccessVectorValues local = ravv.threadLocalSupplier().get();
                    for (int r = 0; r < 2000; r++) {
                        int ord = r % count;
                        VectorFloat<?> v = local.getVector(ord);
                        for (int i = 0; i < dim; i++) {
                            if (v.get(i) != data[ord][i]) mismatches.incrementAndGet();
                        }
                    }
                } catch (InterruptedException ignore) {}
            });
        }
        start.countDown();
        exec.shutdown();
        assertTrue(exec.awaitTermination(30, TimeUnit.SECONDS));
        assertEquals(0, mismatches.get(), "concurrent reads must agree with source data");
    }

    @Test
    void fromFloatsFactoryRoundTrip()
    {
        int count = 8, dim = 5;
        float[][] data = randomData(13, count, dim);
        ByteBufferRandomAccessVectorValues ravv = ByteBufferRandomAccessVectorValues.fromFloats(data, dim);
        assertEquals(count, ravv.size());
        assertEquals(dim, ravv.dimension());
        for (int ord = 0; ord < count; ord++) {
            VectorFloat<?> v = ravv.getVector(ord);
            for (int i = 0; i < dim; i++) assertEquals(data[ord][i], v.get(i), 0f);
        }
    }
}
