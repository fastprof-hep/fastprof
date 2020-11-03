grep 95% $1 > grp1
grep 95% $2 > grp2
diff grp1 grp2 > /dev/null 2>&1
exit $?

