# SPDX-FileCopyrightText: 2021 Aaron Weimann
#
# SPDX-License-Identifier: GPL-3.0-or-later

BEGIN {
    FPAT = "([^,]+)|(\"[^\"]+\")"
}

{
    len = length($4)
    i = substr($4, 2, len - 2) 
    if (i + 0 > threshold){
        len = length($1)
        print substr($1, 2, len - 2)
    } 

}
