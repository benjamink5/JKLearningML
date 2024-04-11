//: # Natural Language Framewok

import NaturalLanguage

//: ## Language Recognizor

//let message = "אין עשן בלי אש"
//let message = "Ich wünsche dir einen guten Morgen!"
//let message = "空穴來風，未必無因"
var message = "안녕하세요"

if let language = NLLanguageRecognizer.dominantLanguage(for: message) {
    print("Dtected \(language.rawValue.uppercased()) as dominant lanuage for: \n\"\(message)\"")
} else {
    print("Could not recognize language for : \(message)")
}

let languageRecognizor = NLLanguageRecognizer()
languageRecognizor.processString(message)
let languateWithProbabilities = languageRecognizor.languageHypotheses(withMaximum: 3)
for (language, probability) in languateWithProbabilities {
    print("Dected \(language.rawValue.uppercased()), probability \(String(format: "%.3f", probability * 100))%)")
}
languageRecognizor.reset()


//:## String Tokenizer
//: This playground project shows how to use NLP to dissect a text into semantic units.
//: Tokenization stands at the core of most other features, so this is an important exercise.
message = "Knowledge will give you power, but character respect."

let tagger = NLTagger(tagSchemes: [NLTagScheme.tokenType])
tagger.string = message

tagger.enumerateTags(in: message.startIndex..<message.endIndex,
                     unit: NLTokenUnit.word,
                     scheme: NLTagScheme.tokenType,
                     options: [.omitPunctuation, .omitWhitespace]) { (tag, range) -> Bool in
    print(message[range])
    
    return true
}

//: ## References
//: - [Apple Documentation - Natural Language](https://developer.apple.com/documentation/naturallanguage)
