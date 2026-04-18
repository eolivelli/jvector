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

import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class MMapRandomAccessVectorValuesTest
{
    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void readsMatchSourceDataWithNoFloatArrayPerCall() throws IOException
    {
        int count = 100, dim = 16;
        Random r = new Random(2026);
        float[][] source = new float[count][dim];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dim; j++) source[i][j] = r.nextFloat() * 2f - 1f;
        }

        File f = tmp.newFile("vectors.bin");
        ByteBuffer bb = ByteBuffer.allocate(count * dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] v : source) for (float x : v) bb.putFloat(x);
        bb.rewind();
        try (RandomAccessFile raf = new RandomAccessFile(f, "rw")) {
            raf.getChannel().write(bb);
        }

        try (MMapRandomAccessVectorValues ravv = new MMapRandomAccessVectorValues(f, dim)) {
            assertEquals(count, ravv.size());
            assertEquals(dim, ravv.dimension());
            for (int ord = 0; ord < count; ord++) {
                VectorFloat<?> v = ravv.getVector(ord);
                for (int i = 0; i < dim; i++) {
                    assertEquals(source[ord][i], v.get(i), 0f);
                }
            }
        }
    }
}
