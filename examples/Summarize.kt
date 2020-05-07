/* Copyright 2019-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuraltokenizer.Sentence as TkSentence
import com.kotlinnlp.summarizer.Summary
import com.sun.xml.internal.messaging.saaj.util.ByteInputStream
import com.xenomachina.argparser.mainBody
import org.apache.tika.metadata.Metadata
import org.apache.tika.parser.AutoDetectParser
import org.apache.tika.sax.BodyContentHandler
import java.io.File
import java.io.StringWriter

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

      val inputText: String = parseDocument(File(filePath).readBytes())
      val (tkSentences, summary) = helper.summarize(inputText)

      printItemsets(summary)

      while (true) {

        println()

        readSummaryStrength()?.also {
          printSummary(tkSentences = tkSentences, summary = summary, summaryStrength = it)
        } ?: break
      }
    }
  }

  println("\nExiting...")
}

/**
 * Read the path of a document from the standard input.
 *
 * @return the path read
 */
private fun readPath(): String {

  print("Insert the path of the document whose content will be summarized (empty to exit): ")

  return readLine()!!
}

/**
 * Parse a document with Apache Tika.
 *
 * @param document a document as array of bytes
 *
 * @return the textual content of the given document
 */
private fun parseDocument(document: ByteArray): String {

  val handler = BodyContentHandler(StringWriter())

  @Suppress("DEPRECATION")
  AutoDetectParser().parse(ByteInputStream(document, document.size), handler, Metadata())

  return handler.toString().replace(Regex("-\n"), "")
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
 * Print the frequent itemsets of a summary.
 *
 * @param summary the summary information
 */
private fun printItemsets(summary: Summary) {

  val sortedItemsets: Sequence<Summary.Itemset> = summary.relevantItemsets.asSequence().sortedByDescending { it.score }

  println()
  println("Relevant itemsets:")
  println(sortedItemsets.joinToString("\n") { "[%5.1f %%] ${it.text}".format(100.0 * it.score) })
}

/**
 * Print a summary of the sentences with a given strength.
 *
 * @param tkSentences the list of tokenized sentences
 * @param summary the summary information
 * @param summaryStrength the percentage of summary (0 means no summary) based on which
 */
private fun printSummary(tkSentences: List<TkSentence>, summary: Summary, summaryStrength: Double) {

  val summarizedText: String = tkSentences.zip(summary.salienceScores)
    .asSequence()
    .filter { it.second >= summaryStrength }
    .map { it.first.buildText() }
    .joinToString("\n------------------------------------------\n")

  println()
  println("Summary:")
  println(summarizedText)
}

