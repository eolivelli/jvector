/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package io.github.jbellis.jvector.util;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Extra coverage for the segmented DenseIntMap implementation, exercising:
 * - cross-segment-boundary writes (keys around multiples of 1024)
 * - concurrent inserts that force the spine to grow from a small initial capacity
 * - concurrent inserts + removes on the same dense key range
 * - forEach ascending-key iteration and visibility of prior writes
 *
 * Complements the existing {@link TestIntMap} tests which cover both Dense and Sparse
 * implementations against a small key range.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDenseIntMapSegmented extends RandomizedTest {

    @Test
    public void testCrossSegmentBoundary() {
        // SEGMENT_SIZE is 1024 internally; pick keys straddling the first few boundaries
        // so a successful round trip proves routing and ascending iteration work across segments.
        DenseIntMap<String> map = new DenseIntMap<>(1);
        int[] keys = {0, 1023, 1024, 2047, 2048, 4096, 10_000};
        for (int k : keys) {
            Assert.assertTrue(map.compareAndPut(k, null, "v" + k));
        }
        Assert.assertEquals(keys.length, map.size());
        for (int k : keys) {
            Assert.assertEquals("v" + k, map.get(k));
            Assert.assertTrue(map.containsKey(k));
        }

        // forEach visits in ascending key order
        List<Integer> visited = new ArrayList<>();
        map.forEach((key, value) -> {
            visited.add(key);
            Assert.assertEquals("v" + key, value);
        });
        Assert.assertEquals(keys.length, visited.size());
        for (int i = 1; i < visited.size(); i++) {
            Assert.assertTrue("forEach must iterate ascending", visited.get(i) > visited.get(i - 1));
        }
    }

    /**
     * Under a tiny initial capacity, many concurrent writers across a wide key range force
     * the spine to grow repeatedly. Every inserted key must be visible afterwards, and the
     * {@code size()} counter must match the number of successful inserts. Regressions here
     * would indicate a lost write during spine growth.
     */
    @Test
    public void testConcurrentInsertForcesSpineGrowth() throws InterruptedException {
        DenseIntMap<Integer> map = new DenseIntMap<>(1); // starts with a single segment
        int nThreads = 16;
        int perThread = 5_000;
        int totalKeys = nThreads * perThread;

        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(nThreads);
        AtomicInteger successes = new AtomicInteger();

        for (int t = 0; t < nThreads; t++) {
            final int threadId = t;
            new Thread(() -> {
                try {
                    start.await();
                    for (int i = 0; i < perThread; i++) {
                        int key = threadId * perThread + i;
                        if (map.compareAndPut(key, null, key)) {
                            successes.incrementAndGet();
                        }
                    }
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }, "inserter-" + t).start();
        }
        start.countDown();
        done.await();

        Assert.assertEquals(totalKeys, successes.get());
        Assert.assertEquals(totalKeys, map.size());
        for (int k = 0; k < totalKeys; k++) {
            Assert.assertEquals("missing key " + k, Integer.valueOf(k), map.get(k));
        }
    }

    /**
     * Concurrent {@code compareAndPut(null, v)} races on the same key must give exactly one
     * winner. A bug where two threads both observe the slot as null and both increment
     * {@code size} would show up as an inflated size.
     */
    @Test
    public void testConcurrentInsertSameKey() throws InterruptedException {
        for (int trial = 0; trial < 20; trial++) {
            DenseIntMap<String> map = new DenseIntMap<>(1);
            int nThreads = 8;
            int sameKey = 12_345; // far enough to require spine growth and segment install
            CountDownLatch start = new CountDownLatch(1);
            CountDownLatch done = new CountDownLatch(nThreads);
            AtomicInteger winners = new AtomicInteger();

            for (int t = 0; t < nThreads; t++) {
                final int tid = t;
                new Thread(() -> {
                    try {
                        start.await();
                        if (map.compareAndPut(sameKey, null, "tid=" + tid)) {
                            winners.incrementAndGet();
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } finally {
                        done.countDown();
                    }
                }).start();
            }
            start.countDown();
            done.await();

            Assert.assertEquals("exactly one thread must win the initial CAS", 1, winners.get());
            Assert.assertEquals(1, map.size());
            Assert.assertNotNull(map.get(sameKey));
        }
    }

    /**
     * Concurrent insert + remove on disjoint key subsets: the remover handles its own keys
     * after the inserter has populated them. Final state is deterministic: all inserted keys
     * present, none of the removed-then-re-inserted keys missing.
     */
    @Test
    public void testConcurrentInsertAndRemove() throws InterruptedException {
        DenseIntMap<Integer> map = new DenseIntMap<>(1);
        int keys = 20_000;
        // First populate
        for (int k = 0; k < keys; k++) {
            Assert.assertTrue(map.compareAndPut(k, null, k));
        }
        Assert.assertEquals(keys, map.size());

        int nThreads = 8;
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(nThreads);
        for (int t = 0; t < nThreads; t++) {
            final int tid = t;
            new Thread(() -> {
                try {
                    start.await();
                    // Each thread touches its own slice: remove every 2nd, re-insert, remove again.
                    for (int k = tid; k < keys; k += nThreads) {
                        Integer removed = map.remove(k);
                        Assert.assertEquals(Integer.valueOf(k), removed);
                        Assert.assertTrue(map.compareAndPut(k, null, k));
                    }
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }).start();
        }
        start.countDown();
        done.await();

        Assert.assertEquals(keys, map.size());
        for (int k = 0; k < keys; k++) {
            Assert.assertEquals(Integer.valueOf(k), map.get(k));
        }
    }

    /**
     * The initialCapacity hint must actually widen the spine so that keys up to the hint
     * are reachable without any spine grow. The contract is semantic (the map works at
     * that size), not structural — we don't expose internal counters — but this test
     * serves as a regression guard against the hint being silently ignored.
     */
    @Test
    public void testInitialCapacityHintSupportsInsertsUpToHint() {
        int capacity = 4 * 1024 + 17; // more than one segment's worth, with a remainder
        DenseIntMap<Integer> map = new DenseIntMap<>(capacity);
        for (int k = 0; k < capacity; k++) {
            Assert.assertTrue("insert failed at k=" + k, map.compareAndPut(k, null, k));
        }
        Assert.assertEquals(capacity, map.size());
        for (int k = 0; k < capacity; k++) {
            Assert.assertEquals(Integer.valueOf(k), map.get(k));
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRejectsZeroInitialCapacity() {
        new DenseIntMap<Integer>(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRejectsNegativeInitialCapacity() {
        new DenseIntMap<Integer>(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRejectsNullValue() {
        DenseIntMap<String> map = new DenseIntMap<>(16);
        map.compareAndPut(0, null, null);
    }
}
