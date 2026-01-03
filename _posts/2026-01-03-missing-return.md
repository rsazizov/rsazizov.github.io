---
layout: post
title: Missing Returns and Undefined Behavior in C++
---

A while back, I ran into some poorly written C++ code and spent quite a bit of time debugging it. Here’s a minimal version:

```c++
#include <iostream>

int function1() {
    std::cout << "function1()\n";
}

void function2() {
    std::cout << "function2()\n";
}

int main() {
    function1();
}
```

An attentive reader might spot a missing return statement in `function1` (which I didn't at the time).
Let's try to compile and run this:

```
$ g++ main.cpp -o main && ./main
main.cpp: In function ‘int function1()’:
main.cpp:5:1: warning: no return statement in function returning non-void [-Wreturn-type]
    5 | }
      | ^
function1()
```

The compiler gives a small warning, but the program seems to work as expected. Let’s see what happens when we compile with `-O1` optimization:

```
main.cpp: In function ‘int function1()’:
main.cpp:5:1: warning: no return statement in function returning non-void [-Wreturn-type]
    5 | }
      | ^
function1()
function2()
function1()
function2()
Segmentation fault (core dumped)
```

Surprisingly, `function2` was executed even though it was never called. To understand what happens, we need to learn
about Undefined Behavior.

## Undefined Behavior

Most low-level languages have scenarios known as undefined behavior (UB). A classic example in C++ is using a variable before initializing it 

```c++
#include <iostream>

int main() {
    int x;
    std::cout << x << std::endl;
    
    return 0;
}
```

Because x is uninitialized, the program's behavior is undefined. It often simply prints whatever "garbage" value that was present at that memory address.
Undefined behavior means that the C++ standard does not define a clear outcome and how the 
program behaves is left entirely up to the compiler and operating system, meaning you cannot rely on it.

Missing a return in a non-void function is another type of undefined behavior. To understand what really happens,
we need to translate our code to assembly.

Here is assembly of code with no optimizations:

```asm
_Z9function1v:
.LFB1731:
    ...
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT ; std::cout << "function1\n"
    ...
	ret ; <-- Return, function ends here
	.cfi_endproc

```

We can see that at the end of the `function1` there is a return instruction emitted by GCC (however since there is no return value, it is still invalid). The ret instruction transfers control back to `main()`. 
Let's see what happens with `-O1`:

```asm
_Z9function1v:
.LFB1812:
    ; ...
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT ; std::cout << "function1\n"
	; Return missing
	.cfi_endproc
...
_Z9function2v:
.LFB1813:
    ...
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT ; std::cout << "function2()\n"
	...
	ret
	.cfi_endproc

```

With -O1, the `ret` instruction disappears entirely, which means that the CPU doesn't jump back to `main()`. Instead, it "falls through" and continues executing the next instructions in memory, which happen to belong to `function2`.

> Note that as we talked earlier, "falling through" is purely accidental here. Since the behavior is undefined, compiler
> can inline, reorder or otherwise transform the code in an unpredictable way leading to unexpected execution paths.

There is actually one interesting exception to this rule which is `main()` function. Unlike other non-void functions, reaching the end of `main()` without a return is allowed in C++. The compiler automatically returns 0, which makes `return` in the main function optional.

## Conclusion

Never ignore GCC warnings. Even better, treat them as errors with `-Wall -Wextra -Werror` flags - this can help catch bugs early on.
