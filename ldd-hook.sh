# shellcheck shell=bash

echo Sourcing checkLddNotFoundsPhase

if [[ ! -v lddFailIfNotFound ]] ; then
    # variable not set, use defaults
    lddFailIfNotFound=1
elif [[ ${lddFailIfNotFound} = 1 ]] ; then
    lddFailIfNotFound=1
else
    lddFailIfNotFound=0
fi

if [[ ! -v lddForcePrint ]] ; then
    # variable not set, use defaults
    lddForcePrint=0
elif [[ ${lddForcePrint} = 1 ]] ; then
    lddForcePrint=1
else
    lddForcePrint=0
fi

echo "checkLddNotFoundsPhase: lddFailIfNotFound=${lddFailIfNotFound}"

checkLddNotFounds() {
    local errors
    errors=0
    if @ldd@ "$1" | grep -q "not found"
    then
        errors=1
    fi

    if (( errors || lddForcePrint )) ; then
        echo checkLddNotFoundsPhase: ldd "$1" >&2
        @ldd@ "$1" >&2

        echo checkLddNotFoundsPhase: patchelf --print-rpath "$1": >&2
        echo -ne '\t' >&2
        @patchelf@ --print-rpath "$1" >&2
    fi

    if (( errors )) ; then
        return 1
    fi
    return 0
}

checkLddNotFoundsPhase() {
    echo Executing checkLddNotFoundsPhase...

    local errors
    errors=0
    for output in $(getAllOutputNames); do
        while IFS= read -r so ; do
            if ! checkLddNotFounds "$so" ; then
                errors=1
            fi
        done < <( find "${!output}" -type f \( -iname '*.so' -or -executable \) )
    done

    if (( errors && lddFailIfNotFound )) ; then
        echo "checkLddNotFoundsPhase: errors detected. Set lddFailIfNotFound = false to ignore" >&2
        exit 1
    elif (( errors && ! lddFailIfNotFound )) ; then
        echo "checkLddNotFoundsPhase: errors detected. Set to ignore" >&2
    fi

    echo Finished executing checkLddNotFoundsPhase...
}

postFixupHooks+=(checkLddNotFoundsPhase)
