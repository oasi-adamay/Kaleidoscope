#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <cctype>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// 翻訳は、https://peta.okechan.net/blog/archives/2888　から引用

using namespace llvm;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
// 未知の文字の場合、字句解析器は0以上255以下のトークン値を返す。
// 既知のトークンなら、そのトークンに合った値を返す。
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5
};

static std::string IdentifierStr; // 現在のトークンがtok_identifierの場合のみ有効　Filled in if tok_identifier
static double NumVal;             // 現在のトークンがtok_numberの場合のみ有効　Filled in if tok_number

// gettok - Return the next token from standard input.
// gettok - 標準入力から次のトークンを返す。
// トークン間の空白を無視する
// 識別子と数値、予約語の認識
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  // 空白をスキップする。
  while (isspace(LastChar))
    LastChar = getchar();

  if (isalpha(LastChar)) { //識別子 identifier: [a-zA-Z][a-zA-Z0-9]*
	//識別子を字句解析したらすぐにグローバル変数IdentifierStrにセットしてる
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { //数値 Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');
	//十分なエラーチェックを行っていない
    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {	//コメント　Comment
    // Comment until end of line.
	// 行の終わりまでがコメント
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  // ファイルの終わりをチェックする。
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  // それ以外の場合には、文字のASCIIコード値をそのまま返す。
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//
namespace {
// ExprAST - Base class for all expression nodes.
// ExprAST - 全ての式ノードの基底クラス。
class ExprAST {
public:
  virtual ~ExprAST() {}
  virtual Value *codegen() = 0;
};

// NumberExprAST - Expression class for numeric literals like "1.0".
// NumberExprAST - "1.0"のような数値リテラルのための式クラス。
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
  Value *codegen() override;
};

// VariableExprAST - Expression class for referencing a variable, like "a".
// VariableExprAST - "a"のような変数を参照するための式クラス。
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}
  Value *codegen() override;
};

// BinaryExprAST - Expression class for a binary operator.
// BinaryExprAST - 二項演算子のための式クラス。
class BinaryExprAST : public ExprAST {
  char Op;		//opcode
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  Value *codegen() override;
};

// CallExprAST - Expression class for function calls.
// CallExprAST - 関数呼び出しのための式クラス。
class CallExprAST : public ExprAST {
  std::string Callee;	//関数名
  std::vector<std::unique_ptr<ExprAST>> Args;	//引数

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}
  Value *codegen() override;
};

// PrototypeAST - This class represents the "prototype" for a function,
// which captures its name, and its argument names (thus implicitly the number
// of arguments the function takes).
// PrototypeAST - 関数のプロトタイプを表すクラス。
// 関数名と引数名をキャプチャする。
// (なので暗黙のうちに引数の数もキャプチャすることになる。)
class PrototypeAST {
  std::string Name;		//関数名
  std::vector<std::string> Args;		//引数名

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args)
      : Name(Name), Args(std::move(Args)) {}
  Function *codegen();
  const std::string &getName() const { return Name; }
};

// FunctionAST - This class represents a function definition itself.
// FunctionAST - 関数定義それ自身を表すクラス。
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;	//ptototype
  std::unique_ptr<ExprAST> Body;		//関数実体

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
  Function *codegen();
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/*
	再帰下降構文解析

	以下のように、ASTを構築したい。
	ExprAST *X = new VariableExprAST("x");
	ExprAST *Y = new VariableExprAST("y");
	ExprAST *Result = new BinaryExprAST('+', X, Y);
*/

// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
// token the parser is looking at.  getNextToken reads another token from the
// lexer and updates CurTok with its results.
// CurTok/getNextToken - シンプルなトークンのバッファを提供する。
// CurTokは構文解析器が現在処理中のトークンである。
static int CurTok;
// getNextTokenは字句解析器から別のトークンを読み取り、その結果でCurTokを更新する。
static int getNextToken() { return CurTok = gettok(); }

// BinopPrecedence - This holds the precedence for each binary operator that is defined.
// BinopPrecedence - 定義された各二項演算子の優先順位を保持する。 
// oprandと優先度のmap　優先度は1が一番低い優先度
static std::map<char, int> BinopPrecedence;

// GetTokPrecedence - Get the precedence of the pending binary operator token.
// GetTokPrecedence - 処理中の二項演算子トークンの優先順位を得る。
// 定義済みの二項演算子でなければ-1を返す。
static int GetTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}

// Error* - These are little helper functions for error handling.
// Error* - これらはエラー処理のための小さな関数群である。
std::unique_ptr<ExprAST> Error(const char *Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> ErrorP(const char *Str) {
  Error(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

// numberexpr ::= number
// 現在のトークンがtok_numberの場合にこの関数が呼び出される事を想定している。
// この関数は現在の数値（NumVal）を読み取り、NumberExprASTノードを生成し、字句解析器を次のトークンへと進め、そして最後にノードを返す。
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = llvm::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

// parenexpr ::= '(' expression ')'
// この関数は、現在のトークンが”(“の場合に呼ばれることを想定している
// 再帰的にParseExpressionを呼び出している。
// 丸括弧そのものはASTノードの構築を行わないことに注意
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();	// expression
  if (!V)
    return nullptr;

  //副次式を解析した後、”)”の出現がない可能性がある。
  if (CurTok != ')')
    return Error("expected ')'");
  getNextToken(); // eat ).
  return V;
}

// identifierexpr
//   ::= identifier
//   ::= identifier '(' expression* ')'
// 変数の参照と関数の呼び出し
// 現在のトークンがtok_identifierの場合に呼び出されることを想定している。
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  //現在の識別子が単なる変数の参照か、関数呼び出しの式のどちらなのかを決定するために先読みする。
  getNextToken(); // eat identifier.

  if (CurTok != '(') // 単なる変数の参照。 Simple variable ref. 
    return llvm::make_unique<VariableExprAST>(IdName);

  // 関数呼び出し。 Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (1) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return Error("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return llvm::make_unique<CallExprAST>(IdName, std::move(Args));
}

// primary
//   ::= identifierexpr
//   ::= numberexpr
//   ::= parenexpr
// ”プライマリ”式
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
  default:
    return Error("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  }
}

// binoprhs
//   ::= ('+' primary)*
// 二項演算式
// 演算子順位構文解析法（Operator-Precedence Parsing）
// ExprPrec: 優先順位は、解析すべき演算子の最小優先順位を表す。
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  // 二項演算子ならその優先順位を得る。
  while (1) {
    int TokPrec = GetTokPrecedence();	//処理中の二項演算式の優先度

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
	// 演算子の優先順位がExprPrecより大きければ処理し、そうでなければ終了する。
	// 無効なトークンは優先順位 - 1になるようにしてるので、トークンの連なりが二項演算子を通り越したら、ペアストリーム（二項演算子とプライマリ式のペアの連なり）が終わった事をこのチェック処理は暗黙的に知ることが出来る。
	if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
	// チェック処理を通ったということは, CurTokは二項演算子である。
    int BinOp = CurTok;		//保存した二項演算子
    getNextToken(); // eat binop

    // Parse the primary expression after the binary operator.
	// 二項演算子の後のプライマリ式を解析する。
    auto RHS = ParsePrimary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
	// BinOpの優先順位がRHS（右手側）の後の二項演算子より低いなら、
	// 処理中の演算子はRHSをそのLHS（左手側）として受け取る。
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
	// LHSとRHSをマージする。
    LHS =
        llvm::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
  }
}

// expression
//   ::= primary binoprhs
//　式は、プライマリ式ではじまりその後に[二項演算子, プライマリ式]（[binop, primaryexpr]）のペアが連なってるものである。　
//	a+b+(c+d)*e*f+g
//  a, [+,b], [+,(c+d)], [*, e], [*,f], [+,g] 
static std::unique_ptr<ExprAST> ParseExpression() {
  //プライマリ式ではじまり...
  auto LHS = ParsePrimary();
  if (!LHS)
    return nullptr;

  //[二項演算子, プライマリ式]が続く。
  return ParseBinOpRHS(0, std::move(LHS));
}

// prototype
//   ::= id '(' id* ')'
// プロトタイプ
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != tok_identifier)
    return ErrorP("Expected function name in prototype");

  std::string FnName = IdentifierStr;
  getNextToken();

  if (CurTok != '(')
    return ErrorP("Expected '(' in prototype");

  // 引数の名前のリストを読み取る。
  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return ErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  return llvm::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

// definition ::= 'def' prototype expression
// 関数本体
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

// toplevelexpr ::= expression
// トップレベル式
// 無名で引数が無い関数として定義する
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
	// 無名のプロトタイプを作成する。
    auto Proto = llvm::make_unique<PrototypeAST>("__anon_expr",
                                                 std::vector<std::string>());
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

// external ::= 'extern' prototype
// external宣言
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static std::unique_ptr<Module> TheModule;	//LLVM IRがコードを収容するために使用するトップレベルの構造物
static IRBuilder<> Builder(getGlobalContext());	//LLVM命令を生成するのを簡単にするためのヘルパオブジェクト
static std::map<std::string, Value *> NamedValues;	//現在のスコープにおける、名前とLLVM Value

Value *ErrorV(const char *Str) {
  Error(Str);
  return nullptr;
}

Value *NumberExprAST::codegen() {
  //LLVM IRでは、定数は、すべてお互いにユニークであり共有される
  return ConstantFP::get(getGlobalContext(), APFloat(Val));
}

Value *VariableExprAST::codegen() {
  // 変数はすでにどこかで発行済みであり、その値が使用できる状態になっていると仮定している。
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    return ErrorV("Unknown variable name");
  return V;
}

Value *BinaryExprAST::codegen() {
  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R)
    return nullptr;
  
  //IRBuilderは、直近で生成した命令をどこに挿入したかを知っているので、
  //どの命令を生成するのか、どのオペランドを使用するのか、を指定するだけでよい。
  //（名前は必須ではない）
  switch (Op) {
  case '+':
    return Builder.CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder.CreateFSub(L, R, "subtmp");
  case '*':
    return Builder.CreateFMul(L, R, "multmp");
  case '<':
    L = Builder.CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
	// boolの0/1をdoubleの0.0/1.0に変換する。
    return Builder.CreateUIToFP(L, Type::getDoubleTy(getGlobalContext()),
                                "booltmp");
  default:
    return ErrorV("invalid binary operator");
  }
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  // // グローバルモジュールテーブルから名前を探す。
  Function *CalleeF = TheModule->getFunction(Callee);
  if (!CalleeF)
    return ErrorV("Unknown function referenced");

  // If argument mismatch error.
  // 引数の数のチェック。
  if (CalleeF->arg_size() != Args.size())
    return ErrorV("Incorrect # arguments passed");

  // 呼び出す関数を取得できたら、その関数に渡される各引数に対して再帰的にコード生成を行う。
  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back())
      return nullptr;
  }

  // 最後に、LLVMのcall命令を生成する。
  return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}

Function *PrototypeAST::codegen() {
  // Make the function type:  double(double,double) etc.
  // 引数の型リストを生成
	std::vector<Type *> Doubles(Args.size(),
                              Type::getDoubleTy(getGlobalContext()));
  // double(double, double)などの関数型を作る。
  FunctionType *FT =
      FunctionType::get(Type::getDoubleTy(getGlobalContext()), Doubles, false);

  //N個のdoubleの引数をとり、それらが可変長（vararg）ではなく（falseがそれを示している）、ひとつのdoubleの戻り値を返す関数型がFunctionType::getによって生成される。
  //TheModuleが引数に指定されているので、NameはTheModuleのシンボルテーブルに登録される。
  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  // 全ての引数について名前をセットする。
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);

  //TODO 引数名の重複をチェックする
  return F;
}

Function *FunctionAST::codegen() {
  // First, check for an existing function from a previous 'extern' declaration.
  // 最初に、以前に、extern宣言された関数が存在するかチェックする。
  Function *TheFunction = TheModule->getFunction(Proto->getName());

  //その関数のプロトタイプ（Proto）のコード生成を行い、戻り値をチェックする
  if (!TheFunction)
    TheFunction = Proto->codegen();

  //プロトタイプのコード生成によって、この後の処理に用いるLLVM関数オブジェクトの存在が確実なものとなる。
  if (!TheFunction)
    return nullptr;

  // Create a new basic block to start insertion into.
  // 挿入するための、新しい基本ブロックを作成する。
  BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);	//新しい基本ブロック（名前は”entry”）が作成され、TheFunctionの中に挿入される。
  Builder.SetInsertPoint(BB);	//今後の命令を新しい基本ブロックの末尾に挿入すべきであるという事Builderに伝える。

  // Record the function arguments in the NamedValues map.
  // 関数引数の名前を保存する。
  NamedValues.clear();
  for (auto &Arg : TheFunction->args())
    NamedValues[Arg.getName()] = &Arg;

  //関数のルート式（root expression）のためにCodeGen()メソッドを呼ぶ。
  //entryブロックの中に、式を計算しその結果を返すためのコードが生成さる。
  if (Value *RetVal = Body->codegen()) {
    // Finish off the function.
	// エラーが起きなければ、関数を完成させるため、LLVMのret命令が生成される。
    Builder.CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
	// 矛盾が無いか、生成されたコードを検証する。
    verifyFunction(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  // 本体のコード生成でエラーがあれば、関数を消去する。
  TheFunction->eraseFromParent();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition:");
      FnIR->dump();
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->dump();
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read top-level expression:");
      FnIR->dump();
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (1) {
    fprintf(stderr, "ready> ");
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  // Make the module, which holds all the code.
  TheModule = llvm::make_unique<Module>("my cool jit", getGlobalContext());

  // Run the main "interpreter loop" now.
  MainLoop();

  // Print out all of the generated code.
  TheModule->dump();

  return 0;
}
