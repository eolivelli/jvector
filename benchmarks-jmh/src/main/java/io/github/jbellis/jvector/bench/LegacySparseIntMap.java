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

import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

/**
 * Verbatim copy of the previous {@code SparseIntMap} implementation
 * ({@link ConcurrentHashMap}{@code <Integer, T>}). Kept in the benchmarks module so
 * {@link SparseIntMapConcurrentBenchmark} can compare the new striped Agrona-backed impl
 * against the old boxing one under identical conditions, without requiring a separate
 * checkout of an older revision.
 * <p>
 * <b>Do not use in production code.</b> Per jvector#3, this representation pays a boxed
 * {@code Integer} per key and a {@code ConcurrentHashMap.Node} per entry — both eliminated
 * by the current production {@code SparseIntMap}.
 */
public class LegacySparseIntMap<T> implements IntMap<T> {
    private final ConcurrentHashMap<Integer, T> map;

    public LegacySparseIntMap() {
        this.map = new ConcurrentHashMap<>();
    }

    @Override
    public boolean compareAndPut(int key, T existing, T value) {
        if (value == null) {
            throw new IllegalArgumentException("compareAndPut() value cannot be null -- use remove() instead");
        }

        if (existing == null) {
            T result = map.putIfAbsent(key, value);
            return result == null;
        }

        return map.replace(key, existing, value);
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public T get(int key) {
        return map.get(key);
    }

    @Override
    public T remove(int key) {
        return map.remove(key);
    }

    @Override
    public boolean containsKey(int key) {
        return map.containsKey(key);
    }

    public IntStream keysStream() {
        return map.keySet().stream().mapToInt(key -> key);
    }

    @Override
    public void forEach(IntBiConsumer<T> consumer) {
        map.forEach(consumer::consume);
    }
}
