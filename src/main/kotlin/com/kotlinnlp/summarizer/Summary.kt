/* Copyright 2019-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.summarizer

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
}
