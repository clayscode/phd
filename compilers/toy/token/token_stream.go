package token

type TokenStream interface {
	Next() bool
	Value() Token
	Peek() Token
}

type SliceTokenStream struct {
	tokens   []Token
	position int
}

func NewSliceTokenStream(tokens []Token) *SliceTokenStream {
	return &SliceTokenStream{tokens: tokens}
}

func (i *SliceTokenStream) Next() bool {
	i.position++
	return i.position <= len(i.tokens)
}

func (i *SliceTokenStream) Value() Token {
	if i.position > len(i.tokens) {
		panic("iterator ended")
	}
	// We increment the position before we get the value, so i.position needs to
	// be negatively offset.
	return i.tokens[i.position-1]
}

func (i *SliceTokenStream) Peek() Token {
	if i.position > len(i.tokens)-1 {
		return Token{Type: EofToken, Value: "EOF"}
	}
	return i.tokens[i.position]
}
