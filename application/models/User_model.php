<?php
class User_model extends CI_Model
{
    var $table = 'user';
    var $column_order = array('fullname', 'email'); //set column field database for datatable orderable
    var $column_search = array('id', 'fullname', 'email'); //set column field database for order and search
    var $order = array('id' => 'desc'); // default order

    public function __construct()
    {
        parent::__construct();
        $this->load->database();
    }

    function get_datatables($role = false)
    {
        $this->_get_datatables_query($role);
        if ($_POST['length'] != -1)
            $this->db->limit($_POST['length'], $_POST['start']);
        $this->db->where('role', $role);
        $query = $this->db->get();
        return $query->result();
    }

    public function getRows()
    {
        $this->db->from($this->table);

        $query = $this->db->get();

        return $query->num_rows();
    }


    private function _get_datatables_query($role)
    {
        $this->db->from($this->table);
        $this->db->where('role', $role);
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

    function count_filtered($role)
    {
        $this->_get_datatables_query($role);
        $query = $this->db->get();
        return $query->num_rows();
    }

    public function count_all($role)
    {
        $this->db->from($this->table);
        $this->db->where('role', $role);
        return $this->db->count_all_results();
    }

    public function get_by_id($id)
    {
        $this->db->from($this->table);
        $this->db->where('id', $id);
        $query = $this->db->get();

        return $query->row();
    }

    public function save($data)
    {
        $this->db->insert($this->table, $data);
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
        $this->db->delete($this->table);
    }

    public function get_user($id = 0)
    {
        if ($id === 0) {
            $query = $this->db->get('user');
            return $query->result_array();
        }

        $query = $this->db->get_where('user', array('id' => $id));
        return $query->row_array();
    }

    public function get_user_login($email, $password)
    {
        $query = $this->db->get_where('user', array('email' => $email, 'password' => md5($password)));
        //return $query->num_rows();
        return $query->row_array();
    }

    public function set_user($id = 0)
    {
        $data = array(
            'fullname' => $this->input->post('fullname'),
            'email' => $this->input->post('email'),
            'password' => $this->input->post('password'),
            'updated_at' => date('Y-m-d H:i:s')
        );

        if ($id == 0) {
            return $this->db->insert('user', $data);
        } else {
            $this->db->where('id', $id);
            return $this->db->update('user', $data);
        }
    }

    public function delete_user($id)
    {
        $this->db->where('id', $id);
        return $this->db->delete('user');
    }
}
