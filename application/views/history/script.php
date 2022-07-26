<script>
    $(document).ready(function() {

        //datatables
        table = $('#table').DataTable({
            "processing": true, //Feature control the processing indicator.
            "serverSide": true, //Feature control DataTables' server-side processing mode.
            "order": [], //Initial no order.


            // Load data for the table's content from an Ajax source
            "ajax": {
                "url": "<?= base_url() ?>history/ajax_list",
                "type": "POST",
            },

            //Set column definition initialisation properties.
            "columnDefs": [{
                    "targets": [0], //first column
                    "orderable": false, //set not orderable
                },
                {
                    "targets": [1], //last column
                    "orderable": false, //set not orderable
                }
            ],
        });

    });

    function reloadTable() {
        table.ajax.reload(null, false); //reload datatable ajax
    }
</script>