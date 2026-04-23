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
import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.management.ManagementFactory;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * Measures the per-entry memory footprint of {@link SparseIntMap} versus the legacy
 * {@link java.util.concurrent.ConcurrentHashMap}{@code <Integer,T>}-backed implementation
 * ({@link LegacySparseIntMap}). The point is to make the heap-savings claim from
 * jvector#3 reproducible without launching a 100M-vector workload.
 * <p>
 * Caveats: heap accounting via {@link ManagementFactory#getMemoryMXBean()} is intrinsically
 * GC-noisy. We trigger {@link System#gc()} before/after the populate step and average over a
 * handful of warmup + measurement iterations; even so, treat the absolute numbers as an
 * upper bound and trust the {@code legacy / striped} ratio rather than any single run.
 * <p>
 * Run with: {@code -bm ss} (single-shot) so each iteration starts from a fresh heap snapshot.
 */
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector",
        "-Xms2g", "-Xmx2g"})
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@State(Scope.Benchmark)
public class SparseIntMapMemoryBenchmark {

    public enum Impl {
        legacy(LegacySparseIntMap::new),
        striped(SparseIntMap::new);

        final Supplier<IntMap<Object>> factory;

        Impl(Supplier<IntMap<Object>> factory) {
            this.factory = factory;
        }
    }

    @Param
    public Impl impl;

    @Param({"100000", "1000000"})
    public int totalKeys;

    /**
     * The map being measured. Held in a field so it isn't GC'd between the
     * populate step and the {@code System.gc()} that captures used-bytes.
     */
    private IntMap<Object> map;

    /** A shared dummy value held by every entry, so the per-entry value cost is constant. */
    private Object value;

    /**
     * Carries the measured heap delta out of the benchmark as JMH secondary metrics. JMH's
     * primary score for a {@link Mode#SingleShotTime} run is execution time; the byte numbers
     * come through these counters in the standard "Secondary result" output.
     */
    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class Counters {
        public long bytesUsed;
        public long bytesPerEntry;

        @Setup(Level.Iteration)
        public void reset() {
            bytesUsed = 0;
            bytesPerEntry = 0;
        }
    }

    @Setup
    public void setup() {
        this.value = new Object();
    }

    @Benchmark
    public long populateAndMeasure(Counters c, Blackhole bh) {
        gcQuiesce();
        long before = usedHeap();

        IntMap<Object> m = impl.factory.get();
        for (int i = 0; i < totalKeys; i++) {
            m.compareAndPut(i, null, value);
        }
        // Pin the map across the GC.
        this.map = m;

        gcQuiesce();
        long after = usedHeap();
        long delta = after - before;
        c.bytesUsed = delta;
        c.bytesPerEntry = m.size() == 0 ? 0 : delta / m.size();
        bh.consume(m.size());

        // Drop the strong reference so the next iteration starts from a clean slate.
        this.map = null;
        return delta;
    }

    private static long usedHeap() {
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }

    private static void gcQuiesce() {
        for (int i = 0; i < 3; i++) {
            System.gc();
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
