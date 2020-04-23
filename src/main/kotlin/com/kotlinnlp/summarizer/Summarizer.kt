/* Copyright 2019-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.summarizer

import ca.pfv.spmf.algorithms.frequentpatterns.lcm.AlgoLCM
import ca.pfv.spmf.algorithms.frequentpatterns.lcm.Dataset
import ca.pfv.spmf.patterns.itemset_array_integers_with_count.Itemset
import com.kotlinnlp.linguisticdescription.morphology.morphologies.ContentWord
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.forEachGroup
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Calculate the salience scores and the relevant itemsets of the sentences that compose a text, with the purpose to
 * build a summary.
 *
 * The algorithm is based on the LSA-itemset summarizer described in:
 *   [Generazione automatica di riassunti di collezioni di documenti multilingua](http://webthesis.biblio.polito.it/id/eprint/6457)
 * that uses the LCM algorithm to extract the frequent itemsets, described in:
 *   [LCM ver. 2: Efficient Mining Algorithms for Frequent/Closed/Maximal Itemsets](http://ceur-ws.org/Vol-126/uno.pdf)
 *
 * @param sentences a list of sentences that compose a text
 * @param ignoreLemmas a blacklist of lemmas that must be ignored when extracting the frequent itemsets
 * @param minLCMSupport the parameter given to the LCM as minimum support value (in percentage respect to the number of
 *                      frequent itemsets collected in the whole text).
 * @param ngramDimRange the range of possible dimensions (number of terms) of an ngram
 */
class Summarizer(
  private val sentences: List<MorphoSynSentence>,
  private val ignoreLemmas: Set<String> = setOf(),
  private val minLCMSupport: Double = 0.01,
  private val ngramDimRange: IntRange = 2 .. 4
) {

  /**
   * The dictionary of relevant terms of the input text.
   */
  private lateinit var termsDictionary: DictionarySet<String>

  /**
   * The dictionary of ngrams (as sequence of relevant terms) of the input text.
   */
  private lateinit var ngramsDictionary: DictionarySet<List<Int>>

  /**
   * The frequent itemsets found in the input text.
   */
  private lateinit var frequentItemsets: List<Itemset>

  /**
   * Check requirements.
   */
  init {
    require(this.sentences.isNotEmpty())
  }

  /**
   * @return the summary of the input text
   */
  fun getSummary(): Summary {

    val sentencesOfLemmas: List<List<String>> = this.sentences.map { this.extractLemmas(it) }
    val itemsetsMatrix: DenseNDArray = this.buildItemsetsMatrix(sentencesOfLemmas)
    val (u, s, v) = itemsetsMatrix.sparseSVD()

    val relevantSingularValues: Int = this.calcRelevantSingularValues(s)
    val itemsetsRelevance: List<Double> =
      this.calcRelevanceScores(s = s, m = u, relevantSingularValues = relevantSingularValues)

    return Summary(
      salienceScores = this.calcRelevanceScores(s = s, m = v, relevantSingularValues = relevantSingularValues),
      relevantItemsets = this.frequentItemsets.zip(itemsetsRelevance)
        .map { (itemset, relevance) -> Summary.Itemset(text = itemset.toText(), score = relevance) }
    )
  }

  /**
   * Calculate the relevance scores respect to an SVD singular matrix.
   *
   * @param s the S matrix of the itemset matrix SVD
   * @param m a singular matrix of the itemset matrix SVD (either U or V)
   * @param relevantSingularValues how many singular values use for the calculation
   *
   * @return the relevance scores of the rows of the given singular matrix
   */
  private fun calcRelevanceScores(s: DenseNDArray, m: DenseNDArray, relevantSingularValues: Int): List<Double> {

    val rowScores: List<Double> = (0 until m.rows).map { k ->

      val sqrScore = (0 .. relevantSingularValues).sumByDouble { i ->

        val mKI: Double = m[k, i]
        val sI: Double = s[i]

        mKI * mKI * sI * sI
      }

      sqrt(sqrScore)
    }

    val maxRowScore: Double = rowScores.max()!!

    return rowScores.map { it / maxRowScore }
  }

  /**
   * @param s the S matrix of the itemset matrix SVD
   *
   * @return the number of relevant singular values
   */
  private fun calcRelevantSingularValues(s: DenseNDArray): Int {

    val singularValuesThreshold: Double = s[0] / 2
    var relevantSingularValues = -1

    // Note: singular values in S are sorted by descending value.
    @Suppress("ControlFlowWithEmptyBody")
    while (relevantSingularValues < s.lastIndex && s[++relevantSingularValues] >= singularValuesThreshold);

    return relevantSingularValues
  }

  /**
   * @param sentence an input sentence
   *
   * @return the list of relevant lemmas that compose the sentence
   */
  private fun extractLemmas(sentence: MorphoSynSentence): List<String> =
    sentence.tokens
      .asSequence()
      .mapNotNull { it.flatMorphologies.firstOrNull() }
      .filter { it is ContentWord && it.lemma !in this.ignoreLemmas }
      .map { it.lemma }
      .toList()

  /**
   * @param sentencesOfTerms a list of sentences (as lists of terms) that compose a text
   *
   * @return the matrix that represents the relation between sentences and frequent itemsets
   */
  private fun buildItemsetsMatrix(sentencesOfTerms: List<List<String>>): DenseNDArray {

    val sentencesOfInt: List<IntArray> = this.sentencesToItems(sentencesOfTerms)

    val dataset = Dataset(sentencesOfInt.filter { it.isNotEmpty() })
    this.frequentItemsets = AlgoLCM().runAlgorithm(this.minLCMSupport, dataset, null).levels.flatten()

    val itemsetsMatrix = DenseNDArrayFactory.zeros(Shape(this.frequentItemsets.size, sentencesOfInt.size))

    sentencesOfInt.forEachIndexed { j, items ->
      this.frequentItemsets.forEachIndexed { i, itemset ->
        if (items.contains(itemset)) itemsetsMatrix[i, j] = 1.0
      }
    }

    return itemsetsMatrix
  }

  /**
   * Convert sentences of terms to ordered sequences of integer items.
   * Each item refers to an ngram of consecutive terms, with a size in the range [ngramDimRange].
   *
   * @param sentencesOfTerms a list of sentences (as lists of terms) that compose a text
   *
   * @return the sentences converted to ordered sequences of integer items
   */
  private fun sentencesToItems(sentencesOfTerms: List<List<String>>): List<IntArray> {

    this.termsDictionary = DictionarySet()
    this.ngramsDictionary = DictionarySet()

    return sentencesOfTerms.map { terms ->

      if (terms.size >= this.ngramDimRange.first) {

        val itemset: MutableSet<Int> = mutableSetOf()
        val intTerms: List<Int> = terms.map { this.termsDictionary.add(it) }

        intTerms.forEachGroup(min = this.ngramDimRange.first, max = this.ngramDimRange.last) { ngram ->
          itemset.add(this.ngramsDictionary.add(ngram))
        }

        itemset.sorted().toIntArray()

      } else {
        intArrayOf()
      }
    }
  }

  /**
   * @param itemset an itemset
   *
   * @return true if this ordered int array contains the given items
   */
  private fun IntArray.contains(itemset: Itemset): Boolean {

    val startIndex: Int = this.indexOf(itemset.items.first())

    if (startIndex < 0) return false

    val endIndex: Int = min(startIndex + itemset.size() - 1, this.lastIndex)

    return this.sliceArray(startIndex .. endIndex).contentEquals(itemset.items)
  }

  /**
   * @return the textual representation of this itemset
   */
  private fun Itemset.toText(): String = this.items.joinToString(", ") { item ->

    val ngram: List<Int> = this@Summarizer.ngramsDictionary.getElement(item)!!
    val terms: List<String> = ngram.map { this@Summarizer.termsDictionary.getElement(it)!! }

    terms.joinToString(" ")
  }
}
