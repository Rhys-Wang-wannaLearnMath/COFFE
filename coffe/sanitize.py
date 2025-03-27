import ast
import re
import traceback
from typing import List, Optional



class CodeVisitor(ast.NodeVisitor):
    def __init__(self, code):
        self.funcs = []
        self.func_names = []
        self.classes = []
        self.code = code
        self.classname = "global"
        self.has_class = False
        self.has_input = False
        self.only_func = False
        self.all_func_in_class = False

    def visit_ClassDef(self, node):
        self.classname = node.name
        self.has_class = True
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classname = "global"

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "input":
            self.has_input = True
        
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.func_names.append("{}@{}".format(node.name, self.classname))
        if self.classname == "global":
            self.funcs.append(node.name)
        self.generic_visit(node)

    def run(self):
        self.root = ast.parse(self.code)
        self.only_func = True
        for statement in self.root.body:
            if not isinstance(statement, ast.FunctionDef) and not isinstance(statement, ast.ClassDef):
                self.only_func = False
                break
        self.visit(self.root)

        if len(self.classes) > 0 and len(self.func_names) > 0:
            self.all_func_in_class = True
            for f in self.func_names:
                if f.split("@")[-1] == "global":
                    self.all_func_in_class = False
                    break

class PlaceHolder(ast.NodeTransformer):
    def __init__(self):
        pass
    
    def visit_FunctionDef(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_AsyncFunctionDef(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node
    
    def visit_ClassDef(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_If(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_For(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node
    
    def visit_While(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_AsyncFor(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_With(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_AsyncWith(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_Try(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_TryStar(self, node):
        if len(node.body) == 0:
            node.body.append(ast.Pass())
            return node
        else:
            self.generic_visit(node)
            return node
    
    def run(self, root):
        self.visit(root)
        return root


class CommentRemover(ast.NodeTransformer):
    def __init__(self, code):
        self.root = ast.parse(code)

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None
    
    def visit_Expr(self, node):
        self.generic_visit(node)
        if type(node.value) == ast.Constant and isinstance(node.value.value, str):
            return None
        else:
            return node
    
    def run(self):
        self.visit(self.root)
        placeholder = PlaceHolder()
        self.root = placeholder.run(self.root)
        ast.fix_missing_locations(self.root)
        new_code = ast.unparse(self.root)
        return new_code




class CodeProcessor(ast.NodeTransformer):
    def __init__(self, code, entry_point = None, force_rename = False):
        self.code = code
        self.entry_point = entry_point
        self.classname = "global"
        self.mode = "funcname"
        self.ori_name = None
        self.force_rename = force_rename


    def visit_ClassDef(self, node):
        self.classname = node.name
        self.generic_visit(node)
        self.classname = "global"

    def visit_Name(self, node):
        if self.mode == "funcname" and self.ori_name != None and node.id == self.ori_name:
            node.id = "solution"
        
        return node


    def visit_FunctionDef(self, node):
        #rename the first function generated as LLMs tend to generate extra useless code in the end of response
        if not self.entry_point and self.mode == "funcname" and node.name == self.visitor.funcs[-1] and self.classname == "global":
            self.ori_name = node.name
            node.name = "solution"
        elif self.entry_point and self.mode == "funcname" and node.name == self.entry_point and self.classname == "global":
            self.ori_name = node.name
            node.name = "solution"

        self.generic_visit(node)
        
        return node

    def visit_Call(self, node):
        if self.mode == "input" and isinstance(node.func, ast.Name) and node.func.id == "input":
            new_node = ast.Name(id = "inputs")
            ast.fix_missing_locations(new_node)
            return new_node
        else:
            self.generic_visit(node)
    

        return node

    def run(self, no_modify = False):
        try:
            remover = CommentRemover(self.code)
            new_code = remover.run().strip()
            if len(new_code) == 0:
                return -1, False
            self.visitor = CodeVisitor(new_code)
            self.visitor.run()
            self.root = ast.parse(self.code)
            if self.visitor.all_func_in_class:
                if no_modify:
                    return ast.unparse(self.root), False
                self.classname = self.visitor.classes[-1]
                self.funcname = None
                for func_name in self.visitor.func_names:
                    if func_name.split("@")[-1] == self.classname:
                        self.funcname = func_name.split("@")[0]
                args = ast.arguments(posonlyargs = [], args = [], vararg = ast.arg(arg = "args"), kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [])
                init_statement = ast.Assign(targets = [ast.Name(id = "s", ctx = ast.Store)], value = ast.Call(func = ast.Name(id = self.classname, ctx = ast.Load), args = [], keywords = []), type_comment = None)
                ast.fix_missing_locations(init_statement)
                call_statement = ast.Expr(value = ast.Call(func = ast.Attribute(value = ast.Name(id = "s", ctx = ast.Load), attr = self.funcname, ctx = ast.Store), args = [ast.Starred(value = ast.Name(id = "args", ctx = ast.Load))], keywords = []))
                ast.fix_missing_locations(call_statement)
                statements = [init_statement, call_statement]
                new_node = ast.FunctionDef(name = "solution", args = args, body = statements, decorator_list =[], returns = None, type_comment = None, type_params = [])
                ast.fix_missing_locations(new_node)
                self.root.body = self.root.body + [new_node]
                return ast.unparse(self.root), False
            elif (len(self.visitor.funcs) > 0 and self.visitor.only_func) or self.force_rename:
                if no_modify:
                    return ast.unparse(self.root), False
                self.mode = "funcname"
                self.visit(self.root)
                return ast.unparse(self.root), False
            else:
                return self.code, True
        except Exception as e:
            #print(e)
            #traceback.print_exc()
            return -1, False
        

    


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def remove_unindented_lines(code, protect_before, execeptions, trim_tails):
    lines = code.splitlines()
    cut_idx = []
    cut_enabled = False
    for i, line in enumerate(lines):
        if not cut_enabled and line.startswith(protect_before):
            cut_enabled = True
            continue
        if line.strip() == "":
            continue
        if any(line.startswith(e) for e in execeptions):
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            cut_idx.append(i)

        if any(line.rstrip().startswith(t) for t in trim_tails):
            # cut off everything behind
            cut_idx.extend(list(range(i, len(lines))))
            break

    return "\n".join([line for i, line in enumerate(lines) if i not in cut_idx])


def to_four_space_indents(old_code):
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code

def remove_space_for_codegen(old_code):
    new_code = ""
    for line in old_code.splitlines():
        if len(line.strip()) == 0:
            new_code += "\n"
            continue
        lspace = len(line) - len(line.lstrip())
        if lspace % 4 == 2:
            new_code += " "*(lspace - 2)
        elif lspace%4 == 0:
            new_code += " "*lspace
        items = line.lstrip().split("  ")
        for index, i in enumerate(items):
            new_code += i.replace(" ", "")
            if index < len(items) - 1:
                new_code += " "
        
        new_code += "\n"

    return new_code
        



def sanitize(
    old_code: str,
    entry_point: str,
    rm_prefix_lines: Optional[str] = None,
    eofs: List = None,
    codegen: bool = False,
    global_code: bool = False,
    chat: bool = False
):
    new_code = old_code.replace("\r\n", "\n").replace("\\_", "_").replace("if __name__", "if 1 or __name__")
    if "<|end_header_id|>" in new_code:
        new_code = new_code.split("<|end_header_id|>")[-1]
    if codegen:
        new_code = remove_space_for_codegen(new_code)
    if global_code and "```" in new_code:
        if len(new_code.split("```python\n")) > 1:
            new_code = new_code.split("```python\n")[1]
        elif len(new_code.split("```")) > 2:
            new_code = new_code.split("```")[1]
        new_code = new_code.split("```")[0]
        
        return new_code.strip()


    if new_code.endswith("```"):
        new_code = new_code[:-3]
    if rm_prefix_lines is not None:
        new_code = "\n".join(
            [
                line
                for line in old_code.splitlines()
                if not line.startswith(rm_prefix_lines)
            ]
        )

    new_code = "\n" + new_code
    def_left = "def " + entry_point

    # basic handling of chat output
    new_code = new_code.replace("\n```python\n", "\n```\n")
    if def_left in new_code:
        for chunk in new_code.split("\n```\n"):
            if def_left in chunk:
                new_code = chunk
                break
    else:
        new_code = new_code.split("```")[0]

    if codegen:
        for chunk in new_code.split("\"\"\""):
            if def_left in chunk:
                new_code = chunk
                break
    
    chunks = [chunk for chunk in re.split(f"{def_left}\s*\(", new_code)]
    # TODO: having return does not mean this is complete
    bodies = [chunk for chunk in chunks[1:] if "    return " in chunk.split("\ndef")[0]]
    def_left = def_left + "("
    new_code = def_left + def_left.join(bodies) if len(bodies) > 0 else ""  # fn + impl
    new_code = to_four_space_indents(new_code)

    for eof in eofs or []:
        new_code = new_code.split(eof)[0]

    # remove lines starting from the first unindented line after def_left
    new_code = remove_unindented_lines(
        new_code,
        protect_before=def_left,
        execeptions=["def ", "import ", "from "],
        trim_tails=['"""', "if", "print"],
    )
    new_code = chunks[0] + new_code

    # cut all functions that are not syntactically correct && not the entry point
    parts = new_code.split("\ndef ")
    includes = [parts[0]]
    for fn in new_code.split("\ndef ")[1:]:
        if (
            fn.strip().startswith(entry_point + " ")
            or fn.strip().startswith(entry_point + "(")
            or syntax_check("\ndef " + fn)
        ):
            includes.append(fn)
    new_code = "\ndef ".join(includes)
    return new_code.strip()