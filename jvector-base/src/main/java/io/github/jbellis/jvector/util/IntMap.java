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

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public interface IntMap<T> {
    /**
     * @param key ordinal
     * @return true if successful, false if the current value != `existing`
     */
    boolean compareAndPut(int key, T existing, T value);

    /**
     * @return number of items that have been added
     */
    int size();

    /**
     * @param key ordinal
     * @return the value of the key, or null if not set
     */
    T get(int key);

    /**
     * @return the former value of the key, or null if it was not set
     */
    T remove(int key);

    /**
     * @return true iff the given key is set in the map
     */
    boolean containsKey(int key);

    /**
     * Iterates over each non-null key/value pair and invokes {@code consumer}.
     * <p>
     * Iteration order is implementation-defined; only {@link DenseIntMap} guarantees ascending
     * keys. Iteration is weakly consistent: entries may be added or removed concurrently and the
     * traversal will reflect the state at some point during the call.
     */
    void forEach(IntBiConsumer<T> consumer);

    /**
     * Iterates over each key currently present in the map and invokes {@code consumer} with
     * the primitive int. Implementations should override to avoid boxing; the default delegates
     * to {@link #forEach(IntBiConsumer)} and discards the value.
     * <p>
     * Iteration order and consistency follow {@link #forEach(IntBiConsumer)}.
     */
    default void forEachKey(IntConsumer consumer) {
        forEach((k, v) -> consumer.accept(k));
    }

    /**
     * Returns a primitive {@link IntStream} of every key currently present in the map.
     * <p>
     * The default builds the stream from {@link #forEachKey(IntConsumer)}. Specialised
     * implementations may override for efficiency. No boxing occurs in either path.
     */
    default IntStream keysStream() {
        IntStream.Builder b = IntStream.builder();
        forEachKey(b::add);
        return b.build();
    }

    @FunctionalInterface
    interface IntBiConsumer<T2> {
        void consume(int key, T2 value);
    }
}
