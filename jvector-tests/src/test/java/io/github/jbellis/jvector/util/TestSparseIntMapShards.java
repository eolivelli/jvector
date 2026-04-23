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
import org.junit.Assert;
import org.junit.Test;

/**
 * White-box tests for {@link SparseIntMap}'s shard distribution. Sequential keys (typical of
 * HNSW node IDs assigned by {@code addGraphNode}) are the worst case for Agrona's identity
 * hashing — without the avalanche mix in {@link SparseIntMap#shardIndex(int)} they would all
 * pile onto one shard.
 */
public class TestSparseIntMapShards extends RandomizedTest {

    @Test
    public void testSequentialKeysSpreadAcrossShards() {
        int n = 100_000;
        int[] perShard = new int[SparseIntMap.SHARD_COUNT];
        for (int i = 0; i < n; i++) {
            perShard[SparseIntMap.shardIndex(i)]++;
        }
        assertWellDistributed(perShard, n);
    }

    @Test
    public void testGappyKeysSpreadAcrossShards() {
        // HNSW upper layers have sparse, gappy node IDs.
        int n = 50_000;
        int[] perShard = new int[SparseIntMap.SHARD_COUNT];
        java.util.Random rnd = new java.util.Random(0xCAFEBABEL);
        for (int i = 0; i < n; i++) {
            perShard[SparseIntMap.shardIndex(rnd.nextInt())]++;
        }
        assertWellDistributed(perShard, n);
    }

    private static void assertWellDistributed(int[] perShard, int total) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int c : perShard) {
            if (c < min) min = c;
            if (c > max) max = c;
        }
        double avg = (double) total / SparseIntMap.SHARD_COUNT;
        // Every shard should hold at least 80% of the average and at most 120%.
        Assert.assertTrue("min shard " + min + " too small (avg=" + avg + ")",
                min >= 0.80 * avg);
        Assert.assertTrue("max shard " + max + " too large (avg=" + avg + ")",
                max <= 1.20 * avg);
    }

    @Test
    public void testShardIndexBoundedAndStable() {
        for (int k : new int[]{0, 1, -1, Integer.MAX_VALUE, Integer.MIN_VALUE, 12345, 0x55555555}) {
            int s = SparseIntMap.shardIndex(k);
            Assert.assertTrue("shard for " + k + " out of range: " + s,
                    s >= 0 && s < SparseIntMap.SHARD_COUNT);
            Assert.assertEquals("shardIndex must be deterministic", s, SparseIntMap.shardIndex(k));
        }
    }
}
