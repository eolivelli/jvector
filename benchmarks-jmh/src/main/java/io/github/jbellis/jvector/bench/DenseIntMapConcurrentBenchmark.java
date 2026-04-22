/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.util.DenseIntMap;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Group;
import org.openjdk.jmh.annotations.GroupThreads;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Measures the throughput of concurrent {@link DenseIntMap} operations — the map that sits on
 * the hot path of {@code ConcurrentNeighborMap} and was identified as the top lock-contention
 * hotspot in a herddb indexing profile.
 * <p>
 * Parameters:
 * <ul>
 *   <li>{@code initialCapacity} — {@code 1024} (default) vs. {@code totalKeys} (pre-sized hint).
 *       The hinted case should show near-zero overhead from internal segment allocation.</li>
 *   <li>{@code totalKeys} — size of the working set. Larger sizes amplify any allocation or
 *       contention overhead.</li>
 * </ul>
 * Thread counts are expressed via {@code @Threads} on each benchmark method: 1 and 8.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@State(Scope.Benchmark)
public class DenseIntMapConcurrentBenchmark {

    @Param({"1024", "1000000"})
    public int initialCapacity;

    @Param({"1000000"})
    public int totalKeys;

    /** Shared map used by the "mixed" (pre-populated) benchmarks. */
    private DenseIntMap<Integer> prepopulated;

    /** Monotonic counter used by insert benchmarks so threads never collide on keys. */
    private final AtomicInteger insertCursor = new AtomicInteger();

    /** The map written to by insert benchmarks. Replaced in each trial's setup. */
    private DenseIntMap<Integer> insertMap;

    @Setup
    public void setup() {
        this.prepopulated = new DenseIntMap<>(initialCapacity);
        for (int i = 0; i < totalKeys; i++) {
            prepopulated.compareAndPut(i, null, i);
        }
        this.insertMap = new DenseIntMap<>(initialCapacity);
        this.insertCursor.set(0);
    }

    /**
     * Models the {@code addNode} insertion pressure during graph build: many threads inserting
     * disjoint dense keys. Under the old RW-lock design this path contended on the read lock
     * whenever any writer happened to be resizing the array.
     */
    @Benchmark
    @Threads(8)
    public boolean insertDense8(Blackhole bh) {
        int key = insertCursor.getAndIncrement();
        if (key >= totalKeys) {
            // Bounded workload; replace the map once we've filled it to avoid unbounded memory growth.
            synchronized (this) {
                if (insertCursor.get() >= totalKeys) {
                    insertMap = new DenseIntMap<>(initialCapacity);
                    insertCursor.set(0);
                }
            }
            key = insertCursor.getAndIncrement();
        }
        boolean ok = insertMap.compareAndPut(key, null, key);
        bh.consume(ok);
        return ok;
    }

    @Benchmark
    @Threads(1)
    public boolean insertDense1(Blackhole bh) {
        return insertDense8(bh);
    }

    /**
     * Models the steady-state {@code insertEdge}/{@code insertDiverse} CAS-update pattern on an
     * already-built base layer: each thread reads then CAS-updates a random pre-populated key.
     */
    @Benchmark
    @Threads(8)
    public boolean casUpdate8(Blackhole bh) {
        int key = ThreadLocalRandom.current().nextInt(totalKeys);
        Integer current = prepopulated.get(key);
        boolean ok = prepopulated.compareAndPut(key, current, key + 1);
        bh.consume(ok);
        return ok;
    }

    @Benchmark
    @Threads(1)
    public boolean casUpdate1(Blackhole bh) {
        return casUpdate8(bh);
    }

    /**
     * Pure {@code get()} throughput under heavy read load — sanity check that the lock-free
     * read path remains as fast as before (and ideally faster, since there is no RW-lock
     * machinery to traverse).
     */
    @Benchmark
    @Threads(8)
    public Integer getHot8() {
        int key = ThreadLocalRandom.current().nextInt(totalKeys);
        return prepopulated.get(key);
    }

    @Benchmark
    @Threads(1)
    public Integer getHot1() {
        return getHot8();
    }

    /**
     * Mixed read/write workload approximating the graph-build steady state: 7 readers for each
     * writer doing a CAS update. Uses JMH groups so both run against the same shared map.
     */
    @Benchmark
    @Group("mixed")
    @GroupThreads(7)
    public Integer mixedRead() {
        int key = ThreadLocalRandom.current().nextInt(totalKeys);
        return prepopulated.get(key);
    }

    @Benchmark
    @Group("mixed")
    @GroupThreads(1)
    public boolean mixedWrite() {
        int key = ThreadLocalRandom.current().nextInt(totalKeys);
        Integer current = prepopulated.get(key);
        return prepopulated.compareAndPut(key, current, key + 1);
    }
}
