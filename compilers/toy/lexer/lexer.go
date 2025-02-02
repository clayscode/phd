package lexer

import (
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"strings"
	"unicode/utf8"
)

type Lexer struct {
	input         string
	startPosition int              // Start of current rune.
	position      int              // Current position in the input.
	width         int              // Width of the last rune read.
	tokens        chan token.Token // Channel of scanned tokens.
	state         stateFunction
}

// Emit a token back to the client.
func (lexer *Lexer) emit(t token.TokenType) {
	lexer.tokens <- token.Token{t, lexer.input[lexer.startPosition:lexer.position]}
	lexer.startPosition = lexer.position
}

// Report an error and exit.
func (lexer *Lexer) errorf(format string, args ...interface{}) stateFunction {
	// Set the text to the error message.
	lexer.tokens <- token.Token{
		token.ErrorToken,
		fmt.Sprintf(format, args...),
	}
	return nil // End the lexing loop.
}

func (lexer *Lexer) run() {
	for state := lexStartState; state != nil; {
		state = state(lexer)
	}
	// No more tokens will be delivered.
	close(lexer.tokens)
}

func (lexer *Lexer) next() rune {
	if lexer.position >= len(lexer.input) {
		lexer.width = 0
		return eofRune
	}
	r, width := utf8.DecodeRuneInString(lexer.input[lexer.position:])
	lexer.width = width
	lexer.position += lexer.width
	return r
}

func (lexer *Lexer) ignore() {
	lexer.startPosition = lexer.position
}

func (lexer *Lexer) Backup() {
	lexer.position -= lexer.width
}

func (lexer *Lexer) peek() rune {
	rune := lexer.next()
	lexer.Backup()
	return rune
}

// accept consumes the next rune if it is from the valid set.
func (lexer *Lexer) accept(valid string) bool {
	if strings.IndexRune(valid, lexer.next()) >= 0 {
		return true
	}
	lexer.Backup()
	return false
}

// acceptRun consumes a run of runes from the valid set.
func (lexer *Lexer) acceptRun(valid string) {
	for strings.IndexRune(valid, lexer.next()) >= 0 {

	}
	lexer.Backup()
}

func (lexer *Lexer) NextToken() token.Token {
	for {
		select {
		case t := <-lexer.tokens:
			return t
		default:
			if lexer.state == nil {
				return token.Token{token.EofToken, ""}
			}
			lexer.state = lexer.state(lexer)
		}
	}
	panic("unreachable!")
}

func Lex(input string) *Lexer {
	return &Lexer{
		input:  input,
		state:  lexStartState,
		tokens: make(chan token.Token, 2), // Two items sufficient.
	}
}
