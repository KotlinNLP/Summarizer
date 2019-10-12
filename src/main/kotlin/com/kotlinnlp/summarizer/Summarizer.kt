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
 */
class Summarizer(private val sentences: List<MorphoSynSentence>, private val ignoreLemmas: Set<String> = setOf()) {

  companion object {

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
  }

  /**
   * The dictionary of relevant lemmas of the input text.
   */
  private lateinit var lemmasDictionary: DictionarySet<String>

  /**
   * The dictionary of ngrams (as sequence of relevant lemmas) of the input text.
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

      Math.sqrt(sqrScore)
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
   * @param sentencesOfLemmas a list of sentences (as lists of lemmas) that compose a text
   *
   * @return the matrix that represents the relation between sentences and frequent itemsets
   */
  private fun buildItemsetsMatrix(sentencesOfLemmas: List<List<String>>): DenseNDArray {

    val sentencesOfInt: List<IntArray> = this.sentencesToItems(sentencesOfLemmas)

    val dataset = Dataset(sentencesOfInt.filter { it.isNotEmpty() })
    this.frequentItemsets = AlgoLCM().runAlgorithm(MIN_LCM_SUPPORT, dataset, null).levels.flatten()

    val itemsetsMatrix = DenseNDArrayFactory.zeros(Shape(this.frequentItemsets.size, sentencesOfInt.size))

    sentencesOfInt.forEachIndexed { j, items ->
      this.frequentItemsets.forEachIndexed { i, itemset ->
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

    this.lemmasDictionary = DictionarySet()
    this.ngramsDictionary = DictionarySet()

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

  /**
   *
   */
  private fun Itemset.toText(): String = this.items.joinToString(", ") { item ->

    val ngram: List<Int> = this@Summarizer.ngramsDictionary.getElement(item)!!
    val lemmas: List<String> = ngram.map { this@Summarizer.lemmasDictionary.getElement(it)!! }

    lemmas.joinToString(" ")
  }
}
