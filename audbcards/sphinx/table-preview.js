// Expand row in Tables table to show preview of table content.
//
// Implementation based on https://github.com/chhikaradi1993/Expandable-table-row
//
const toggleRow = (row) => {
    row.getElementsByClassName('expanded-row-content')[0].classList.toggle('hide-row');
    if (row.className.indexOf("clicked") === -1) {
        row.classList.add("clicked");
    } else {
        row.classList.remove("clicked");
    }
    console.log(event);
}
