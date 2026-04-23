/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.util;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.function.Supplier;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestIntMap extends RandomizedTest {

    @Test
    public void testInsert() {
        testInsertInternal(new DenseIntMap<>(100));
        testInsertInternal(new SparseIntMap<>());
    }

    private void testInsertInternal(IntMap<String> map) {
        for (int i = 0; i < 3; i++) {
            Assert.assertNull(map.get(i));
            Assert.assertFalse(map.containsKey(i));

            map.compareAndPut(i, null, "value" + i);
            Assert.assertEquals("value" + i, map.get(i));
            Assert.assertTrue(map.containsKey(i));
            Assert.assertEquals(i + 1, map.size());
        }
    }

    @Test
    public void testUpdate() {
        testUpdateInternal(new DenseIntMap<>(100));
        testUpdateInternal(new SparseIntMap<>());
    }

    private void testUpdateInternal(IntMap<String> map) {
        for (int i = 0; i < 3; i++) {
            map.compareAndPut(i, null, "value" + i);
        }
        Assert.assertEquals(3, map.size());

        for (int i = 0; i < 3; i++) {
            map.compareAndPut(i, map.get(i), "new-value" + i);
            Assert.assertEquals("new-value" + i, map.get(i));
            Assert.assertEquals(3, map.size());
        }
    }

    @Test
    public void testRemove() {
        testRemoveInternal(new DenseIntMap<>(100));
        testRemoveInternal(new SparseIntMap<>());
    }

    private void testRemoveInternal(IntMap<String> map) {
        for (int i = 0; i < 3; i++) {
            map.compareAndPut(i, null, "value" + i);
        }
        Assert.assertEquals(3, map.size());

        for (int i = 0; i < 3; i++) {
            map.remove(i);
            Assert.assertNull(map.get(i));
            Assert.assertFalse(map.containsKey(i));
            Assert.assertEquals(3 - (i + 1), map.size());
        }
    }

    @Test
    public void testForEach() {
        for (Supplier<IntMap<String>> factory : factories()) {
            IntMap<String> map = factory.get();
            int[] keys = {0, 7, 99, 1024, 12345};
            for (int k : keys) {
                map.compareAndPut(k, null, "v" + k);
            }
            Set<Integer> seen = new HashSet<>();
            map.forEach((k, v) -> {
                Assert.assertEquals("v" + k, v);
                Assert.assertTrue("duplicate key " + k, seen.add(k));
            });
            Set<Integer> expected = new HashSet<>();
            for (int k : keys) expected.add(k);
            Assert.assertEquals(expected, seen);
        }
    }

    @Test
    public void testForEachKey() {
        for (Supplier<IntMap<String>> factory : factories()) {
            IntMap<String> map = factory.get();
            int[] keys = {0, 7, 99, 1024, 12345, 99999};
            for (int k : keys) {
                map.compareAndPut(k, null, "v" + k);
            }
            Set<Integer> seen = new HashSet<>();
            map.forEachKey(k -> Assert.assertTrue("duplicate key " + k, seen.add(k)));
            Set<Integer> expected = new HashSet<>();
            for (int k : keys) expected.add(k);
            Assert.assertEquals(expected, seen);
        }
    }

    @Test
    public void testKeysStream() {
        for (Supplier<IntMap<String>> factory : factories()) {
            IntMap<String> map = factory.get();
            int[] keys = {0, 7, 99, 1024, 12345, 99999};
            for (int k : keys) {
                map.compareAndPut(k, null, "v" + k);
            }
            int[] streamed = map.keysStream().sorted().toArray();
            int[] expected = keys.clone();
            java.util.Arrays.sort(expected);
            Assert.assertArrayEquals(expected, streamed);
        }
    }

    @Test
    public void testForEachKeyEmpty() {
        for (Supplier<IntMap<String>> factory : factories()) {
            IntMap<String> map = factory.get();
            map.forEachKey(k -> Assert.fail("empty map should not visit any keys"));
            Assert.assertEquals(0, map.keysStream().count());
        }
    }

    @Test
    public void testCompareAndPutContract() {
        for (Supplier<IntMap<String>> factory : factories()) {
            IntMap<String> map = factory.get();

            // (null, v) on absent → success
            Assert.assertTrue(map.compareAndPut(42, null, "v1"));
            Assert.assertEquals("v1", map.get(42));

            // (null, v) on present → failure
            Assert.assertFalse(map.compareAndPut(42, null, "v2"));
            Assert.assertEquals("v1", map.get(42));

            // (mismatched, v) → failure (different reference even if equals)
            String wrong = new String("v1");
            Assert.assertNotSame("v1", wrong);
            Assert.assertFalse(map.compareAndPut(42, wrong, "v3"));
            Assert.assertEquals("v1", map.get(42));

            // (matching, v') → success
            Assert.assertTrue(map.compareAndPut(42, map.get(42), "v3"));
            Assert.assertEquals("v3", map.get(42));

            // (_, null) → IllegalArgumentException
            try {
                map.compareAndPut(42, map.get(42), null);
                Assert.fail("expected IllegalArgumentException for null value");
            } catch (IllegalArgumentException expected) {
                // OK
            }
        }
    }

    @SuppressWarnings("unchecked")
    private Supplier<IntMap<String>>[] factories() {
        return new Supplier[]{
                () -> new DenseIntMap<String>(100),
                () -> new SparseIntMap<String>()
        };
    }

    @Test
    public void testConcurrency() throws InterruptedException {
        for (int i = 0; i < 100; i++) {
            testConcurrencyOnce(new DenseIntMap<>(100));
            testConcurrencyOnce(new SparseIntMap<>());
        }
    }

    private static void testConcurrencyOnce(IntMap<String> map) throws InterruptedException {
        var source = new ConcurrentHashMap<Integer, String>();

        int nThreads = randomIntBetween(2, 16);
        var latch = new CountDownLatch(nThreads);
        for (int t = 0; t < nThreads; t++) {
            new Thread(() -> {
                try {
                    for (int i = 0; i < 1000; i++) {
                        int key = randomIntBetween(0, 100);
                        if (rarely()) {
                            source.remove(key);
                            map.remove(key);
                        } else {
                            String value = randomAsciiAlphanumOfLength(20);
                            source.put(key, value);
                            map.compareAndPut(key, map.get(key), value);
                        }
                    }
                } finally {
                    latch.countDown();
                }
            }).start();
        }
        latch.await();

        Assert.assertEquals(source.size(), map.size());
        source.forEach((key, value) -> {
            Assert.assertTrue(map.containsKey(key));
            Assert.assertEquals(value, map.get(key));
        });
    }
}
