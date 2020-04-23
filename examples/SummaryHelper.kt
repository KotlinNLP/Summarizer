/* Copyright 2019-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.labelerselector.MorphoSelector
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.neuraltokenizer.Sentence
import com.kotlinnlp.summarizer.Summarizer
import com.kotlinnlp.summarizer.Summary
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileInputStream

/**
 * A helper that summarizes a text and prints progress information.
 *
 * @param parsedArgs the parsed command line arguments
 */
internal class SummaryHelper(parsedArgs: CommandLineArguments) {

  /**
   * A neural tokenizer.
   */
  private val tokenizer: NeuralTokenizer = parsedArgs.tokenizerModelPath.let {
    println("Loading tokenizer model from '$it'...")
    NeuralTokenizer(NeuralTokenizerModel.load(FileInputStream(File(it))))
  }

  /**
   * A neural LHR parser.
   */
  private val parser: LHRParser = parsedArgs.parserModelPath.let {
    println("Loading parser model from '$it'...")
    LHRParser(LHRModel.load(FileInputStream(File(it))))
  }

  /**
   * A morphological analyzer.
   */
  private val analyzer = MorphologicalAnalyzer(dictionary = parsedArgs.morphoDictionaryPath.let {
    println("Loading serialized dictionary from '$it'...")
    MorphologyDictionary.load(FileInputStream(File(it)))
  })

  /**
   * A blacklist of lemmas that must be ignored when extracting the frequent itemsets.
   */
  private val lemmasBlacklist: Set<String> = parsedArgs.lemmasBlacklistPath?.let {
    println("Loading lemmas blacklist for from '$it'...")
    File(it).readLines().toSet()
  } ?: setOf()

  /**
   * Summarize a text.
   *
   * @param text a text
   *
   * @return the list of tokenized sentences and the related summary
   */
  fun summarize(text: String): Pair<List<Sentence>, Summary> {

    val timer = Timer()

    val tkSentences: List<Sentence> = this.tokenize(text)
    val parsedSentences: List<MorphoSynSentence> = this.parse(tkSentences)

    println("Elapsed time: ${timer.formatElapsedTime()}")
    timer.reset()

    println("Summarizing...")
    val summary: Summary = Summarizer(sentences = parsedSentences, ignoreLemmas = this.lemmasBlacklist).getSummary()

    println("Elapsed time: ${timer.formatElapsedTime()}")

    return tkSentences to summary
  }

  /**
   * Tokenize a text.
   *
   * @param text a text
   *
   * @return the list of tokenized sentences
   */
  private fun tokenize(text: String): List<Sentence> {

    println("Tokenizing sentences...")

    val tkSentences: List<Sentence> = this.tokenizer.tokenize(text).filter { it.tokens.isNotEmpty() }

    println("${tkSentences.size} non-empty sentences found.")

    return tkSentences
  }

  /**
   * Parse a list of sentences.
   *
   * @param tkSentences a list of tokenized sentences
   *
   * @return the sentences parsed
   */
  private fun parse(tkSentences: List<Sentence>): List<MorphoSynSentence> {

    println("Parsing sentences...")

    val progress = ProgressIndicatorBar(tkSentences.size)

    return tkSentences.map { sentence ->

      @Suppress("UNCHECKED_CAST")
      val parsingSentence = ParsingSentence(
        tokens = sentence.tokens.mapIndexed { i, token ->
          ParsingToken(id = i, form = token.form, position = token.position)
        },
        position = sentence.position,
        labelerSelector = MorphoSelector,
        morphoAnalysis = this.analyzer.analyze(sentence as RealSentence<RealToken>)
      )

      progress.tick()

      this.parser.parse(parsingSentence)
    }
  }
}
