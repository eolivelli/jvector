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

import io.github.jbellis.jvector.util.IntMap;
import io.github.jbellis.jvector.util.SparseIntMap;
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
import java.util.function.Supplier;

/**
 * Measures the throughput of concurrent {@link SparseIntMap} operations. The map sits on the
 * hot path of {@code ConcurrentNeighborMap} for HNSW layers above the base — every
 * {@code addNode}, {@code insertEdge}, {@code get}/{@code containsKey} during search, and
 * {@code keysStream}/{@code forEachKey} during traversal goes through it.
 * <p>
 * Parameters:
 * <ul>
 *   <li>{@code impl} — {@code legacy} (previous {@link java.util.concurrent.ConcurrentHashMap
 *       ConcurrentHashMap}{@code <Integer,T>}) vs {@code striped} (current striped Agrona
 *       {@code Int2ObjectHashMap} shards). Both implementations run in the same JVM under
 *       identical conditions so results are directly comparable.</li>
 *   <li>{@code keyDensity} — {@code dense} (sequential keys, the worst case for Agrona's
 *       identity hashing) vs {@code sparse} (random keys over a 100x-larger space, mimicking
 *       upper-layer HNSW node IDs).</li>
 *   <li>{@code totalKeys} — size of the working set.</li>
 * </ul>
 * Thread counts are expressed via {@code @Threads} on each benchmark method: 1 and 8.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@State(Scope.Benchmark)
public class SparseIntMapConcurrentBenchmark {

    public enum Impl {
        legacy(LegacySparseIntMap::new),
        striped(SparseIntMap::new);

        final Supplier<IntMap<Integer>> factory;

        Impl(Supplier<IntMap<Integer>> factory) {
            this.factory = factory;
        }
    }

    public enum KeyDensity { dense, sparse }

    @Param
    public Impl impl;

    @Param
    public KeyDensity keyDensity;

    @Param({"100000", "1000000"})
    public int totalKeys;

    /** Pre-populated map used by get/CAS/forEach benchmarks. */
    private IntMap<Integer> prepopulated;

    /** Keys that exist in {@link #prepopulated} (random access benchmarks pick from these). */
    private int[] livekeys;

    /** Monotonic counter used by insert benchmarks so threads never collide on keys. */
    private final AtomicInteger insertCursor = new AtomicInteger();

    /** The map written to by insert benchmarks. Replaced once exhausted. */
    private IntMap<Integer> insertMap;

    @Setup
    public void setup() {
        this.prepopulated = impl.factory.get();
        this.livekeys = new int[totalKeys];
        java.util.Random rnd = new java.util.Random(0xCAFEBABEL);
        for (int i = 0; i < totalKeys; i++) {
            int k = (keyDensity == KeyDensity.dense) ? i : rnd.nextInt(totalKeys * 100);
            livekeys[i] = k;
            prepopulated.compareAndPut(k, null, i);
        }
        this.insertMap = impl.factory.get();
        this.insertCursor.set(0);
    }

    @Benchmark
    @Threads(8)
    public Integer getHot8() {
        int idx = ThreadLocalRandom.current().nextInt(livekeys.length);
        return prepopulated.get(livekeys[idx]);
    }

    @Benchmark
    @Threads(1)
    public Integer getHot1() {
        int idx = ThreadLocalRandom.current().nextInt(livekeys.length);
        return prepopulated.get(livekeys[idx]);
    }

    @Benchmark
    @Threads(8)
    public boolean casChurn8(Blackhole bh) {
        return doCasChurn(bh);
    }

    @Benchmark
    @Threads(1)
    public boolean casChurn1(Blackhole bh) {
        return doCasChurn(bh);
    }

    private boolean doCasChurn(Blackhole bh) {
        int idx = ThreadLocalRandom.current().nextInt(livekeys.length);
        int k = livekeys[idx];
        Integer cur = prepopulated.get(k);
        boolean ok = prepopulated.compareAndPut(k, cur, idx);
        bh.consume(ok);
        return ok;
    }

    /** Models the upper-layer {@code addNode} pressure: many threads inserting disjoint keys. */
    @Benchmark
    @Threads(8)
    public boolean insertSequential8(Blackhole bh) {
        return doInsert(bh);
    }

    @Benchmark
    @Threads(1)
    public boolean insertSequential1(Blackhole bh) {
        return doInsert(bh);
    }

    private boolean doInsert(Blackhole bh) {
        int key = insertCursor.getAndIncrement();
        if (key >= totalKeys) {
            synchronized (this) {
                if (insertCursor.get() >= totalKeys) {
                    insertMap = impl.factory.get();
                    insertCursor.set(0);
                }
            }
            key = insertCursor.getAndIncrement();
        }
        boolean ok = insertMap.compareAndPut(key, null, key);
        bh.consume(ok);
        return ok;
    }

    /**
     * Iteration cost — measured single-threaded since the production callers
     * ({@code OnHeapGraphIndex.nodeStream}) walk it from one thread at a time.
     */
    @Benchmark
    @Threads(1)
    public void forEachKey(Blackhole bh) {
        prepopulated.forEachKey((int k) -> bh.consume(k));
    }

    /** 90 % reads + 10 % CAS-updates: closest to HNSW build's upper-layer steady state. */
    @Benchmark
    @Group("mixed90r10w")
    @GroupThreads(7)
    public Integer mixedRead() {
        int idx = ThreadLocalRandom.current().nextInt(livekeys.length);
        return prepopulated.get(livekeys[idx]);
    }

    @Benchmark
    @Group("mixed90r10w")
    @GroupThreads(1)
    public boolean mixedWrite() {
        int idx = ThreadLocalRandom.current().nextInt(livekeys.length);
        int k = livekeys[idx];
        Integer cur = prepopulated.get(k);
        return prepopulated.compareAndPut(k, cur, idx);
    }
}
