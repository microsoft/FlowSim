#!/usr/bin/env bash

# Exit on unset variables and errors in pipelines, but allow error capture
set -uo pipefail

PROJECT_ROOT=$(dirname "$(realpath "$0")")/..
LINT_CPP=false
LINT_PYTHON=false
DRY_RUN=false

usage() {
    echo "Usage: $0 [cpp|py|dry]"
    echo "  cpp     Lint C++ code only"
    echo "  py      Lint Python code only"
    echo "  dry     Dry run (check only, don't auto-fix)"
}

# Parse argument
if [ $# -eq 1 ]; then
    case "$1" in
        cpp)
            LINT_CPP=true
            ;;
        py)
            LINT_PYTHON=true
            ;;
        dry)
            DRY_RUN=true
            LINT_CPP=true
            LINT_PYTHON=true
            ;;
        *)
            usage
            exit 1
            ;;
    esac
elif [ $# -eq 0 ]; then
    LINT_CPP=true
    LINT_PYTHON=true
else
    usage
    exit 1
fi

exit_code=0

if $LINT_CPP; then
    echo "Linting C++ files..."
    files=$(git ls-files --cached | grep -E '\.(c|h|cpp|hpp|cc|cu|cuh)$' || true)
    if [ -n "$files" ]; then
        echo "Found C++ files to lint:"
        echo "$files"
        if $DRY_RUN; then
            echo "Running clang-format check..."
            clang-format -style=file --dry-run -Werror $files || {
                echo "clang-format found formatting issues in the above files."
                exit_code=1
            }
        else
            echo "Running clang-format auto-fix..."
            clang-format -style=file -i $files
            echo "C++ formatting applied."
        fi
    else
        echo "No C++ files found to lint."
    fi
fi

if $LINT_PYTHON; then
    echo "Linting Python files..."
    files=$(git ls-files --cached | grep -E '\.py$' || true)
    if [ -n "$files" ]; then
        echo "Found Python files to lint:"
        echo "$files"
        if $DRY_RUN; then
            echo "Running black check..."
            python3 -m black --config "${PROJECT_ROOT}/pyproject.toml" --check --diff $files || {
                echo "black found formatting issues in the above files."
                exit_code=1
            }
        else
            echo "Running black auto-fix..."
            python3 -m black --config "${PROJECT_ROOT}/pyproject.toml" $files
            echo "Python formatting applied."
        fi
    else
        echo "No Python files found to lint."
    fi
fi

if [ "$exit_code" -eq 0 ]; then
    echo "Linting completed successfully."
else
    echo "Linting failed. Please fix the formatting issues above."
fi

exit $exit_code
