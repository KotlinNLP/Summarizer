/* Copyright 2019-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.summarizer

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.pow

/**
 * The information of summary of a text.
 *
 * @property salienceScores the salience scores of the sentences that compose the text
 * @property relevantItemsets the relevant itemsets of the text
 */
data class Summary(val salienceScores: List<Double>, val relevantItemsets: List<Itemset>) {

  /**
   * A relevant itemset of the text.
   *
   * @property text the text of the itemset
   * @property score the relevance score
   */
  data class Itemset(val text: String, val score: Double)

  /**
   * A relevant keyword of the text.
   *
   * @property keyword the keyword
   * @property score the relevance score
   */
  data class ScoredKeyword(val keyword: String, val score: Double)

  /**
   * The relevant keywords in the text.
   */
  val relevantKeywords: List<ScoredKeyword> = this.buildRelevantKeywords()

  /**
   * Get the distribution of the salience scores in buckets of a fixed interval.
   * E.g. with 20 buckets: [0, 0.05], [0.05, 0.10], [0.10, 0.15], etc..
   *
   * @param buckets the number of buckets (default 10)
   *
   * @return the distribution of the salience scores in buckets
   */
  fun getSalienceDistribution(buckets: Int = 10): List<Double> {

    val total: Int = this.salienceScores.size
    val counts = IntArray(buckets) { 0 }

    this.salienceScores.forEach { score ->
      val index: Int = max(0, ceil(score * buckets).toInt() - 1)
      counts[index]++
    }

    return List(buckets) { i -> counts[i].toDouble() / total }
  }

  /**
   * @return the list of scored relevant keywords
   */
  private fun buildRelevantKeywords(): List<ScoredKeyword> {

    val keywords: MutableMap<String, MutableList<Double>> = mutableMapOf()

    this.relevantItemsets.forEach { itemset ->
      itemset.text
        .replace(",", " ")
        .replace("  ", " ")
        .split(" ")
        .forEach { keyword -> keywords.getOrPut(keyword) { mutableListOf() }.add(itemset.score) }
    }

    return keywords
      .map { (keyword, scores) ->
        ScoredKeyword(keyword = keyword, score = (scores.average().pow(1.0 / scores.size)))
      }
      .sortedByDescending { it.score }
  }
}
