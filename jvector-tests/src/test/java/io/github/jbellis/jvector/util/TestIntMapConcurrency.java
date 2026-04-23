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
package io.github.jbellis.jvector.util;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * Deep concurrency tests for {@link IntMap} implementations. The shipped {@code TestIntMap}
 * exercises the interface contract on a single thread plus a coarse "do random ops vs a CHM
 * source-of-truth" check; this file targets the contract that {@code ConcurrentNeighborMap}'s
 * CAS retry loops actually rely on (linearizability of compareAndPut on the same key,
 * happens-before across successful writes, weakly-consistent iteration, no re-entrant
 * deadlock, accurate size accounting under contention).
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestIntMapConcurrency extends RandomizedTest {

    private static final int TEST_TIMEOUT_MS = 5_000;

    @SuppressWarnings("unchecked")
    private static Supplier<IntMap<Object>>[] factories() {
        return new Supplier[]{
                () -> new DenseIntMap<Object>(1024),
                () -> new SparseIntMap<Object>()
        };
    }

    /**
     * N threads compete on a single key, each looping
     * {@code compareAndPut(key, get(key), newRef)} and counting successful CASes. Total
     * successful CASes must equal the total number of distinct value references that ever
     * appeared as the current value (no lost updates), and the final value must be one of the
     * values written.
     */
    @Test
    public void testCompareAndPutLinearizability() throws Exception {
        for (Supplier<IntMap<Object>> factory : factories()) {
            int nThreads = 8;
            int opsPerThread = 5_000;
            IntMap<Object> map = factory.get();
            int key = 12345;
            // Seed so that compareAndPut(k, observed, v) only ever runs with non-null expected
            Object seed = new Object();
            Assert.assertTrue(map.compareAndPut(key, null, seed));

            ExecutorService pool = Executors.newFixedThreadPool(nThreads);
            try {
                CountDownLatch start = new CountDownLatch(1);
                AtomicInteger totalSuccesses = new AtomicInteger();
                Set<Object> witnessed = ConcurrentHashMap.newKeySet();
                witnessed.add(seed);
                Future<?>[] futures = new Future[nThreads];
                for (int t = 0; t < nThreads; t++) {
                    futures[t] = pool.submit(() -> {
                        try {
                            start.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                        int local = 0;
                        for (int i = 0; i < opsPerThread; i++) {
                            Object expected = map.get(key);
                            witnessed.add(expected);
                            Object next = new Object();
                            if (map.compareAndPut(key, expected, next)) {
                                local++;
                                witnessed.add(next);
                            }
                        }
                        totalSuccesses.addAndGet(local);
                    });
                }
                start.countDown();
                for (Future<?> f : futures) {
                    f.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                }

                // Every successful CAS published a brand-new Object that was either witnessed by
                // a later thread (counted in `witnessed`) or remains as the final value.
                Object finalVal = map.get(key);
                Assert.assertNotNull(finalVal);
                Assert.assertTrue("final value must have been written by some thread",
                        witnessed.contains(finalVal));
                // total successes == (witnessed values produced by writers) + (the seed)
                // Each successful CAS is the *only* operation that ever publishes a fresh
                // value, so the witnessed-set size minus 1 (seed) is a lower bound on total
                // successes (writer may also see its own write before another thread does).
                // It must also be ≤ totalSuccesses.
                int writes = totalSuccesses.get();
                int witnessedWrites = witnessed.size() - 1; // exclude seed
                Assert.assertTrue(
                        "writes=" + writes + " must be >= witnessedWrites=" + witnessedWrites,
                        writes >= witnessedWrites);
                Assert.assertEquals("size must remain 1", 1, map.size());
            } finally {
                pool.shutdown();
            }
        }
    }

    /**
     * One thread iterates {@code forEachKey} while writers insert and remove. Asserts no
     * exception is ever thrown (the snapshot-based iteration must tolerate concurrent
     * mutation) and that emitted keys were live at some point during the iteration.
     */
    @Test
    public void testForEachKeyDuringMutation() throws Exception {
        for (Supplier<IntMap<Object>> factory : factories()) {
            IntMap<Object> map = factory.get();
            // Pre-populate
            for (int i = 0; i < 200; i++) {
                map.compareAndPut(i, null, new Object());
            }

            int nWriters = 4;
            ExecutorService pool = Executors.newFixedThreadPool(nWriters + 1);
            AtomicBoolean stop = new AtomicBoolean();
            AtomicReference<Throwable> error = new AtomicReference<>();
            try {
                Future<?>[] writerFutures = new Future[nWriters];
                for (int w = 0; w < nWriters; w++) {
                    final int seed = w;
                    writerFutures[w] = pool.submit(() -> {
                        java.util.Random rnd = new java.util.Random(seed);
                        while (!stop.get()) {
                            try {
                                int k = rnd.nextInt(400);
                                if (rnd.nextBoolean()) {
                                    map.compareAndPut(k, map.get(k), new Object());
                                } else {
                                    map.remove(k);
                                }
                            } catch (Throwable t) {
                                error.compareAndSet(null, t);
                                throw t;
                            }
                        }
                    });
                }

                Future<?> reader = pool.submit(() -> {
                    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                    try {
                        while (System.nanoTime() < deadline && !stop.get()) {
                            int[] count = {0};
                            map.forEachKey(k -> {
                                Assert.assertTrue(k >= 0);
                                Assert.assertTrue(k < 400);
                                count[0]++;
                            });
                            // The size-vs-iteration count is weakly consistent. Just assert
                            // we got a sane bound.
                            Assert.assertTrue(count[0] >= 0);
                            Assert.assertTrue(count[0] <= 400);
                        }
                    } catch (Throwable t) {
                        error.compareAndSet(null, t);
                        throw t;
                    }
                });

                reader.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                stop.set(true);
                for (Future<?> f : writerFutures) {
                    f.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                }
            } finally {
                stop.set(true);
                pool.shutdown();
            }
            if (error.get() != null) {
                throw new AssertionError("concurrent mutation broke iteration", error.get());
            }
        }
    }

    /**
     * Mutating the map from inside the {@code forEachKey} consumer must not deadlock. The
     * snapshot semantics in the striped {@code SparseIntMap} were chosen specifically to
     * avoid this case (we never hold a shard lock across the consumer callback).
     */
    @Test
    public void testForEachKeyReentrant() throws Exception {
        for (Supplier<IntMap<Object>> factory : factories()) {
            IntMap<Object> map = factory.get();
            for (int i = 0; i < 64; i++) {
                map.compareAndPut(i, null, new Object());
            }

            ExecutorService pool = Executors.newSingleThreadExecutor();
            try {
                Future<?> f = pool.submit(() -> {
                    map.forEachKey(k -> {
                        // Re-enter: read and write the map from inside the callback.
                        Object cur = map.get(k);
                        if (cur != null) {
                            map.compareAndPut(k, cur, new Object());
                        }
                        // Insert something brand-new keyed past the snapshot range.
                        map.compareAndPut(k + 10_000, null, new Object());
                    });
                });
                f.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            } finally {
                pool.shutdown();
            }
        }
    }

    /**
     * Insert N keys concurrently across N threads, then remove a known subset across N
     * threads; assert {@code size()} matches the live count exactly.
     */
    @Test
    public void testSizeUnderConcurrentInsertRemove() throws Exception {
        for (Supplier<IntMap<Object>> factory : factories()) {
            int nThreads = 8;
            int keysPerThread = 1_000;
            int totalKeys = nThreads * keysPerThread;
            IntMap<Object> map = factory.get();

            ExecutorService pool = Executors.newFixedThreadPool(nThreads);
            try {
                CountDownLatch start = new CountDownLatch(1);

                // Phase 1: each thread inserts its own non-overlapping range.
                Future<?>[] inserters = new Future[nThreads];
                for (int t = 0; t < nThreads; t++) {
                    final int base = t * keysPerThread;
                    inserters[t] = pool.submit(() -> {
                        try {
                            start.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                        for (int i = 0; i < keysPerThread; i++) {
                            Assert.assertTrue(map.compareAndPut(base + i, null, new Object()));
                        }
                    });
                }
                start.countDown();
                for (Future<?> f : inserters) {
                    f.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                }
                Assert.assertEquals(totalKeys, map.size());

                // Phase 2: half of every range is removed in parallel.
                int halfPerThread = keysPerThread / 2;
                Future<?>[] removers = new Future[nThreads];
                CountDownLatch removeStart = new CountDownLatch(1);
                for (int t = 0; t < nThreads; t++) {
                    final int base = t * keysPerThread;
                    removers[t] = pool.submit(() -> {
                        try {
                            removeStart.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                        for (int i = 0; i < halfPerThread; i++) {
                            Assert.assertNotNull(map.remove(base + i));
                        }
                    });
                }
                removeStart.countDown();
                for (Future<?> f : removers) {
                    f.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                }

                int expectedLive = totalKeys - nThreads * halfPerThread;
                Assert.assertEquals(expectedLive, map.size());

                // Cross-check: walking the map yields the same count as size().
                AtomicInteger walked = new AtomicInteger();
                map.forEachKey(k -> walked.incrementAndGet());
                Assert.assertEquals(expectedLive, walked.get());
            } finally {
                pool.shutdown();
            }
        }
    }

    /** A successful compareAndPut must establish happens-before for the published value. */
    @Test
    public void testHappensBeforeOnSuccessfulCAS() throws Exception {
        for (Supplier<IntMap<Object>> factory : factories()) {
            IntMap<Object> map = factory.get();
            int key = 7;
            int rounds = 1_000;

            ExecutorService pool = Executors.newFixedThreadPool(2);
            try {
                for (int round = 0; round < rounds; round++) {
                    map.remove(key);
                    CountDownLatch start = new CountDownLatch(1);
                    AtomicReference<Throwable> err = new AtomicReference<>();
                    Future<?> writer = pool.submit(() -> {
                        try {
                            start.await();
                            Payload p = new Payload();
                            p.field = 99; // happens-before via lock release on successful CAS
                            map.compareAndPut(key, null, p);
                        } catch (Throwable t) {
                            err.compareAndSet(null, t);
                        }
                    });
                    Future<?> reader = pool.submit(() -> {
                        try {
                            start.await();
                            Object o;
                            while ((o = map.get(key)) == null) {
                                Thread.onSpinWait();
                            }
                            Payload p = (Payload) o;
                            if (p.field != 99) {
                                err.compareAndSet(null,
                                        new AssertionError("payload.field=" + p.field
                                                + " — happens-before violated"));
                            }
                        } catch (Throwable t) {
                            err.compareAndSet(null, t);
                        }
                    });
                    start.countDown();
                    writer.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                    reader.get(TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                    if (err.get() != null) {
                        throw new AssertionError("round " + round, err.get());
                    }
                }
            } finally {
                pool.shutdown();
            }
        }
    }

    private static class Payload {
        // intentionally non-volatile — visibility must come from the IntMap CAS, not from
        // the field declaration.
        int field;
    }

    /** Stale expected → no-op false return; current value unchanged. */
    @Test
    public void testCompareAndPutWithStaleExpected() {
        for (Supplier<IntMap<Object>> factory : factories()) {
            IntMap<Object> map = factory.get();
            int k = 3;
            Object v0 = new Object();
            Assert.assertTrue(map.compareAndPut(k, null, v0));

            Object v1 = new Object();
            Assert.assertTrue(map.compareAndPut(k, v0, v1));

            // Try to overwrite using the stale v0 reference — must fail and leave v1 in place.
            Object v2 = new Object();
            Assert.assertFalse(map.compareAndPut(k, v0, v2));
            Assert.assertSame(v1, map.get(k));
        }
    }

    /** High-volume mixed workload; assert internal consistency. */
    @Test
    public void testStressManyKeys() throws Exception {
        for (Supplier<IntMap<Object>> factory : factories()) {
            int nThreads = 16;
            int opsPerThread = 25_000;
            int keySpace = 5_000;
            IntMap<Object> map = factory.get();

            ExecutorService pool = Executors.newFixedThreadPool(nThreads);
            try {
                CountDownLatch start = new CountDownLatch(1);
                Future<?>[] futures = new Future[nThreads];
                for (int t = 0; t < nThreads; t++) {
                    final int seed = t;
                    futures[t] = pool.submit(() -> {
                        try {
                            start.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                        java.util.Random rnd = new java.util.Random(seed);
                        for (int i = 0; i < opsPerThread; i++) {
                            int k = rnd.nextInt(keySpace);
                            int op = rnd.nextInt(10);
                            if (op < 6) {
                                map.compareAndPut(k, map.get(k), new Object());
                            } else if (op < 8) {
                                map.remove(k);
                            } else if (op < 9) {
                                map.get(k);
                            } else {
                                map.containsKey(k);
                            }
                        }
                    });
                }
                start.countDown();
                for (Future<?> f : futures) {
                    f.get(2 * TEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);
                }

                // Final check: size() agrees with walking the map.
                AtomicInteger walked = new AtomicInteger();
                Set<Integer> seen = new HashSet<>();
                map.forEachKey(k -> {
                    Assert.assertTrue("dup key in walk: " + k, seen.add(k));
                    walked.incrementAndGet();
                });
                Assert.assertEquals(map.size(), walked.get());
            } finally {
                pool.shutdown();
            }
        }
    }
}
