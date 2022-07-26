<?php
class Plat_model extends CI_Model
{
    var $table = 'view_plat_nomor';
    var $column_order = array('id_user', 'plat_nomor', 'fullname', 'email'); //set column field database for datatable orderable
    var $column_search = array('id_user', 'plat_nomor', 'fullname', 'email'); //set column field database for order and search
    var $order = array('fullname' => 'asc'); // default order

    public function __construct()
    {
        parent::__construct();
        $this->load->database();
    }

    function get_datatables($sessId = false)
    {
        $this->_get_datatables_query($sessId);
        if ($_POST['length'] != -1)
            $this->db->limit($_POST['length'], $_POST['start']);
        if ($sessId) {
            $this->db->where('id_user', $sessId);
        }
        $query = $this->db->get();
        return $query->result();
    }


    public function getRows()
    {
        $this->db->from($this->table);

        $query = $this->db->get();

        return $query->num_rows();
    }

    private function _get_datatables_query($sessId = false)
    {
        $this->db->from($this->table);
        if ($sessId) {
            $this->db->where('id_user', $sessId);
        }
        $i = 0;

        foreach ($this->column_search as $item) // loop column 
        {
            if ($_POST['search']['value']) // if datatable send POST for search
            {

                if ($i === 0) // first loop
                {
                    $this->db->group_start(); // open bracket. query Where with OR clause better with bracket. because maybe can combine with other WHERE with AND.
                    $this->db->like($item, $_POST['search']['value']);
                } else {
                    $this->db->or_like($item, $_POST['search']['value']);
                }

                if (count($this->column_search) - 1 == $i) //last loop
                    $this->db->group_end(); //close bracket
            }
            $column[$i] = $item; // set column array variable to order processing
            $i++;
        }

        if (isset($_POST['order'])) // here order processing
        {
            $this->db->order_by($column[$_POST['order']['0']['column']], $_POST['order']['0']['dir']);
        } else if (isset($this->order)) {
            $order = $this->order;
            $this->db->order_by(key($order), $order[key($order)]);
        }
    }

    function count_filtered($sessId = false)
    {
        $this->_get_datatables_query($sessId);
        $query = $this->db->get();
        return $query->num_rows();
    }

    public function count_all($sessId = false)
    {
        $this->db->from($this->table);
        if ($sessId) {
            $this->db->where('id_user', $sessId);
        }
        return $this->db->count_all_results();
    }

    public function get_by_id($id, $id_user = false)
    {
        $this->db->from($this->table);
        $this->db->where('id_plat', $id);
        if ($id_user) {
            $this->db->where('id_user', $id_user);
        }
        $query = $this->db->get();

        return $query->row();
    }

    public function get_by_plat_nomor($id)
    {
        $this->db->from($this->table);
        $this->db->where('plat_nomor', $id);
        $query = $this->db->get();

        return $query->num_rows();
    }

    public function save($data)
    {
        $this->db->insert('tbl_plat_nomor', $data);
        return $this->db->insert_id();
    }

    public function update($where, $data)
    {
        $this->db->update($this->table, $data, $where);
        return $this->db->affected_rows();
    }

    public function delete_by_id($id)
    {

        $this->db->where('id', $id);
        $this->db->delete('tbl_plat_nomor');
    }

    public function editStatus($where, $data)
    {
        $this->db->update($this->table, $data, $where);
        return $this->db->affected_rows();
    }
}
