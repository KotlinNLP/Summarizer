/* Copyright 2019-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
internal class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The file path of the serialize model of the NeuralTokenizer.
   */
  val tokenizerModelPath: String by parser.storing(
    "-t",
    "--tokenizer-model-path",
    help="the file path of the serialize model of the NeuralTokenizer"
  )

  /**
   * The file path of the serialized model of the LHR parser model.
   */
  val parserModelPath: String by parser.storing(
    "-p",
    "--parser-model-path",
    help="the file path of the serialized model of the LHR parser model"
  )

  /**
   * The file path of the serialized morphology dictionary.
   */
  val morphoDictionaryPath: String by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the serialized morphology dictionary"
  )

  /**
   * The file path of the lemmas blacklist.
   */
  val lemmasBlacklistPath: String? by parser.storing(
    "-b",
    "--blacklist",
    help="the file path of the lemmas blacklist"
  ).default { null }

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
