/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
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
import com.kotlinnlp.neuraltokenizer.Sentence as TkSentence
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.summarizer.Summarizer
import com.kotlinnlp.summarizer.Summary
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Summarize the text contained in a given file.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val helper = SummaryHelper(parsedArgs = CommandLineArguments(args))

  while (true) {

    println()
    val filePath: String = readPath()

    if (filePath.isEmpty()) {

      break

    } else {

      val inputText: String = File(filePath).readText()

      val summarizedSentences: List<Pair<TkSentence, Double>> = helper.summarize(inputText)
      val scoredSentences: List<Pair<String, Double>> = summarizedSentences.map { it.first.buildText() to it.second }

      while (true) {
        println()
        readSummaryStrength()?.also { printSummary(scoredSentences = scoredSentences, summaryStrength = it) } ?: break
      }
    }
  }

  println("\nExiting...")
}

/**
 * Read the path of a textual file from the standard input.
 *
 * @return the path read
 */
private fun readPath(): String {

  print("Insert the path of the file whose content will be summarized (empty to exit): ")

  return readLine()!!
}

/**
 * Read the summary strength from the standard input.
 *
 * @return the summary strength or null if the input was empty
 */
private fun readSummaryStrength(): Double? {

  print("Insert the summary strength [0.0 - 1.0] (empty to skip to the next file): ")

  return readLine()!!.let { if (it.isNotEmpty()) it.toDouble() else null }
}

/**
 * Print a summary of the sentences with a given strength.
 *
 * @param scoredSentences the list of sentences with the related salience score
 * @param summaryStrength the percentage of summary (0 means no summary) based on which
 */
private fun printSummary(scoredSentences: List<Pair<String, Double>>, summaryStrength: Double) {

  val summary: String = scoredSentences
    .asSequence()
    .filter { it.second >= summaryStrength }
    .map { it.first }
    .joinToString("\n------------------------------------------\n")

  println()
  println("Summary:")
  println(summary)
}

/**
 * A helper that summarizes a text and prints progress information.
 *
 * @param parsedArgs the parsed command line arguments
 */
private class SummaryHelper(parsedArgs: CommandLineArguments) {

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
   * Summarize a text associating a salience score to each sentence that compose it.
   *
   * @param text a text
   *
   * @return a list of sentences associated to the related salience score
   */
  fun summarize(text: String): List<Pair<TkSentence, Double>> {

    val timer = Timer()

    val tkSentences: List<TkSentence> = this.tokenize(text)
    val parsedSentences: List<MorphoSynSentence> = this.parse(tkSentences)

    println("Elapsed time: ${timer.formatElapsedTime()}")
    timer.reset()

    println("Summarizing...")
    val summary: Summary = Summarizer.getSummary(parsedSentences)

    println("Elapsed time: ${timer.formatElapsedTime()}")

    return tkSentences.zip(summary.salienceScores)
  }

  /**
   * Tokenize a text.
   *
   * @param text a text
   *
   * @return the list of tokenized sentences
   */
  private fun tokenize(text: String): List<TkSentence> {

    println("Tokenizing sentences...")

    val tkSentences: List<TkSentence> = this.tokenizer.tokenize(text).filter { it.tokens.isNotEmpty() }

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
  private fun parse(tkSentences: List<TkSentence>): List<MorphoSynSentence> {

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
