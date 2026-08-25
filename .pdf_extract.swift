import Foundation
import PDFKit

let path = CommandLine.arguments[1]
guard let document = PDFDocument(url: URL(fileURLWithPath: path)) else {
    fputs("Unable to open PDF\n", stderr)
    exit(1)
}

for index in 0..<document.pageCount {
    print("=== PAGE \(index + 1) ===")
    print(document.page(at: index)?.string ?? "")
}
