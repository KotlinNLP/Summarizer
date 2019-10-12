/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
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
import kotlin.math.min

/**
 * Helper that calculates the salience scores of sentences that compose a text, with the purpose to build a summary.
 *
 * The algorithm is based on the LSA-itemset summarizer described in:
 *   [Generazione automatica di riassunti di collezioni di documenti multilingua](http://webthesis.biblio.polito.it/id/eprint/6457)
 * that uses the LCM algorithm to extract the frequent itemsets, described in:
 *   [LCM ver. 2: Efficient Mining Algorithms for Frequent/Closed/Maximal Itemsets](http://ceur-ws.org/Vol-126/uno.pdf)
 */
object Summarizer {

  /**
   * The parameter given to the LCM as minimum support value (in percentage respect to the number of frequent itemsets
   * collected in the whole text).
   */
  private const val MIN_LCM_SUPPORT = 0.01

  /**
   * The minimum number of lemmas that compose an ngram.
   */
  private const val MIN_NGRAM_SIZE = 2

  /**
   * The maximum number of lemmas that compose an ngram.
   */
  private const val MAX_NGRAM_SIZE = 4

  /**
   * @param sentences a list of sentences that compose a text
   *
   * @return the salience scores of the input sentences
   */
  fun getSalienceScores(sentences: List<MorphoSynSentence>): List<Double> {

    require(sentences.isNotEmpty())

    val sentencesOfLemmas: List<List<String>> = sentences.map { this.extractLemmas(it) }
    val itemsetsMatrix: DenseNDArray = this.buildItemsetsMatrix(sentencesOfLemmas)
    val (_, s, v) = itemsetsMatrix.sparseSVD()

    // Note: singular values in S are sorted by descending value.
    val singularValuesThreshold: Double = s[0] / 2
    var relevantSingularValues = -1
    while (relevantSingularValues < s.lastIndex && s[++relevantSingularValues] >= singularValuesThreshold);

    val rowScores: List<Double> = (0 until v.rows).map { k ->
      val sqrScore = (0 .. relevantSingularValues).sumByDouble { i -> Math.pow(v[k, i], 2.0) * Math.pow(s[i], 2.0) }
      Math.sqrt(sqrScore)
    }
    val maxRowScore: Double = rowScores.max()!!

    return rowScores.map { it / maxRowScore }
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
      .filter { it is ContentWord }
      .map { it.lemma }
      .toList()

  /**
   * @param sentencesOfLemmas a list of sentences (as lists of lemmas) that compose a text
   *
   * @return the matrix that represents the relation between sentences and frequent itemsets
   */
  private fun buildItemsetsMatrix(sentencesOfLemmas: List<List<String>>): DenseNDArray {

    val sentencesOfInt: List<IntArray> = this.sentencesToItems(sentencesOfLemmas)

    val dataset = Dataset(sentencesOfInt.filter { it.isNotEmpty() })
    val frequentItemsets: List<Itemset> = AlgoLCM().runAlgorithm(MIN_LCM_SUPPORT, dataset, null).levels.flatten()
    val itemsetsMatrix = DenseNDArrayFactory.zeros(Shape(frequentItemsets.size, sentencesOfInt.size))

    sentencesOfInt.forEachIndexed { j, items ->
      frequentItemsets.forEachIndexed { i, itemset ->
        if (items.contains(itemset)) itemsetsMatrix[i, j] = 1.0
      }
    }

    return itemsetsMatrix
  }

  /**
   * Convert sentences of lemmas to ordered sequences of integer items.
   * Each item refers to an ngram of consecutive lemmas, with a size between [MIN_NGRAM_SIZE] and [MAX_NGRAM_SIZE].
   *
   * @param sentencesOfLemmas a list of sentences (as lists of lemmas) that compose a text
   *
   * @return the sentences converted to ordered sequences of integer items
   */
  private fun sentencesToItems(sentencesOfLemmas: List<List<String>>): List<IntArray> {

    val lemmasDictionary = DictionarySet<String>()
    val ngramsDictionary = DictionarySet<List<Int>>()

    return sentencesOfLemmas.map { sentence ->

      val intSentence: List<Int> = sentence.map { lemma ->
        lemmasDictionary.add(lemma)
        lemmasDictionary.getId(lemma)!!
      }
      val sentenceItems: MutableSet<Int> = mutableSetOf()

      (MIN_NGRAM_SIZE .. MAX_NGRAM_SIZE).forEach { ngramSize ->

        val startIter: Int = min(ngramSize, sentence.size)

        (startIter until sentence.size).forEach { endExclusive ->
          val start: Int = endExclusive - ngramSize
          val ngram: List<Int> = intSentence.subList(start, endExclusive)
          ngramsDictionary.add(ngram)
          sentenceItems.add(ngramsDictionary.getId(ngram)!!)
        }
      }

      sentenceItems.sorted().toIntArray()
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
}
